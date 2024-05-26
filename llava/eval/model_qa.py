import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import logging

from llava.conversation import conv_mistral_instruct
from llava.utils import disable_torch_init


@torch.inference_mode()
def eval_model(model_name, questions_file, answers_file):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.float16).cuda()
    if not "mistral" in model_name.lower():
        logging.warning("This model is not a Mistral model. The model may not work as expected. (especially the conversation template)")

    ques_file = open(os.path.expanduser(questions_file), "r")
    ans_file = open(os.path.expanduser(answers_file), "w")
    for i, line in enumerate(tqdm(ques_file)):
        idx = json.loads(line)["question_id"]
        qs = json.loads(line)["text"]
        conv = conv_mistral_instruct.copy()
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        output_ids = model.generate(
            input_ids,
            do_sample=False,
            use_cache=True,
            # temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,  # explicitly request open-ended generation (suppresses warnings)
            max_new_tokens=1024,)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # try:
        #     index = outputs.index(conv.sep, len(prompt))
        # except ValueError:
        #     outputs += conv.sep
        #     index = outputs.index(conv.sep, len(prompt))

        outputs = outputs[len(prompt) + 1:]
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    args = parser.parse_args()

    eval_model(args.model_name, args.question_file, args.answers_file)
