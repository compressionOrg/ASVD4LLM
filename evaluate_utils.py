import torch
import torch.nn as nn
from tqdm import tqdm
import os

from datautils import get_eval_loaders
from lm_eval.base import BaseLM
from lm_eval import evaluator
from datasets import load_dataset
import time
import re
import numpy as np


class EvalLM(BaseLM):
    def __init__(
        self,
        model,
        tokenizer,
        # device="cuda:0",
        batch_size=1,
    ):
        super().__init__()

        # assert isinstance(device, str)
        assert isinstance(batch_size, int)

        # self._device = torch.device(device)
        self._device = model.device

        # self.model = model.to(self.device)
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        self.seqlen = 2048

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0][:, :, :50257]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False)


@torch.no_grad()
def evaluate_perplexity(model, dataset, limit):
    """
    dataset: input ids tensor of shape [batch, sequence length]
    """
    nsamples, seqlen = dataset.size()

    nlls = []

    for i in range(nsamples):
        if i == limit:
            break
        input_ids = dataset[i : i + 1, :-1].to(model.device)
        labels = dataset[i : i + 1, 1:].contiguous()
        logits = model(input_ids=input_ids)[0]
        shift_logits = logits[:, :, :]
        shift_labels = labels.to(model.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
    return ppl.item()


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    model_name,
    tasks,
    eval_ppl="",
    num_fewshot=0,
    limit=-1,
    batch_size=4,
    use_bos=False,
    device="cuda"
):
    """
    model: model name
    limit: number of test samples for debug, set to -1 is no limit
    tasks: str tasks are split by ,
    num_fewshot: Number of examples in few-shot context
    eval_ppl: str datasets are split by , such as 'wikitext2,ptb,c4'
    """
    lm = EvalLM(model, tokenizer, batch_size=batch_size)
    results = {}
    if eval_ppl:
        ppls = {}
        for dataset in eval_ppl.split(","):
            test_loader = get_eval_loaders(dataset, tokenizer, seq_len=2048, batch_size = batch_size)
            nlls = []
            for batch in tqdm(test_loader):
                batch = batch.to(device)
                output = model(batch, use_cache=False)
                lm_logits = output.logits
                if torch.isfinite(lm_logits).all():
                    shift_logits = lm_logits[:, :-1, :].contiguous()
                    shift_labels = batch[:, 1:].contiguous()
                    
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    nlls.append(loss)
            ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
            ppls[dataset] = ppl
        print("PPL after pruning: {}".format(ppls))
        print("Weight Memory: {} GB\n".format(torch.cuda.memory_allocated()/1024/1024/1024))
        
    if tasks == "longbench":
        from tools.eval_longbench import eval_longbench, full_longeval_datasets, small_longeval_datasets

        longbench_results = eval_longbench(model, tokenizer, model_name, datasets=full_longeval_datasets)
        results.update(longbench_results)
        tasks = ""
    elif tasks == "small_longbench":
        from tools.eval_longbench import eval_longbench, full_longeval_datasets, small_longeval_datasets

        longbench_results = eval_longbench(model, tokenizer, model_name, datasets=small_longeval_datasets)
        results.update(longbench_results)
        tasks = ""
    elif tasks == "mmlu":
        tasks = "hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions"
    elif tasks == "llmqat":
        # tasks = "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"
        tasks = "lambada_openai,openbookqa"
    if tasks != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=tasks.split(","),
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=None if limit == -1 else limit,
            no_cache=True,
        )
        t_results = t_results["results"]
        acc_list = [t_results[key]["acc"] for key in t_results.keys() if "acc" in t_results[key]]
        mean_acc = sum(acc_list) / len(acc_list)
        t_results["mean"] = mean_acc
        results.update(t_results)
        print(results)
        print("\n" + "="*50)
        print("EVALUATION RESULTS (formatted for easy copying)")
        print("="*50)
        
        for task_name in sorted(t_results.keys()):
            if task_name != "mean" and "acc" in t_results[task_name]:
                acc_value = t_results[task_name]["acc"] * 100  
                print(f"{task_name}: {acc_value:.2f}%")
        print(f"mean: {mean_acc * 100:.2f}%")  

    return results
