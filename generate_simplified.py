import os
import sys
import argparse
import gradio as gr
import torch
import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from LLMPruner.peft import PeftModel

#from utils.callbacks import Iteratorize, Stream
#from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch_version = int(torch.__version__.split('.')[1])

def main(args):
    if args.model_type == 'pretrain':
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            low_cpu_mem_usage=True if torch_version >=9 else False
        )
        description = "Model Type: {}\n Base Model: {}".format(args.model_type, args.base_model)
    elif args.model_type == 'pruneLLM':
        pruned_dict = torch.load(args.ckpt, map_location='cpu')
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
        description = "Model Type: {}\n Pruned Model: {}".format(args.model_type, args.ckpt)
    elif args.model_type == 'tune_prune_LLM':
        pruned_dict = torch.load(args.ckpt, map_location='cpu')
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
        model = PeftModel.from_pretrained(
            model,
            args.lora_ckpt,
            torch_dtype=torch.float16,
        )
        description = "Model Type: {}\n Pruned Model: {}\n LORA ckpt: {}".format(args.model_type, args.ckpt, args.lora_ckpt)
    else:
        raise NotImplementedError

    if device == "cuda":
        model.half()
        model = model.cuda()
    
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()

    def evaluate(
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        inputs = tokenizer(input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                do_sample=True,
                top_k=50,
                top_p=top_p,
                temperature=temperature,
                max_length=max_new_tokens,
                return_dict_in_generate=True,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output

    output_text = evaluate(args.input_text)
    print(output_text)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLaMA (huggingface version)')

    parser.add_argument('--base_model', type=str, default="liuhaotian/llava-v1.5-7b", help='base model name')
    parser.add_argument('--model_type', type=str, default='pruneLLM', help = 'choose from ')
    parser.add_argument('--input_text', type=str, default='Tell me a funny joke', help = 'Text input for model evaluation')
    parser.add_argument('--ckpt', type=str, default='/shared-local/aoq609/LLM-Pruner/LLMPruner/prune_log/llava-v1.5-7b_0.2/pytorch_model.bin')
    parser.add_argument('--lora_ckpt', type=str, default=None)
    parser.add_argument('--share_gradio', action='store_true')

    args = parser.parse_args()
    main(args)


