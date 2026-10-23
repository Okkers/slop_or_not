import torch 
import pandas as pd 
import numpy as np 
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from vis.utils import construct_wan_from_text
import json

###
# file for creating custom dataset, based on the Persuade Dataset from Kaggle
###

system_prompt = "Respond to the following task to the best of your ability. Output only the essay text, no title or summary."

def infer_MGT_instance(prompts, model, tokenizer, batch_size=4, max_new_tokens=700,
                       temperature=0.7, top_p=0.9):

    chat_prompts = []
    for p in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": p}
        ]
        chat_prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    outs = []
    for s in range(0, len(chat_prompts), batch_size):
        batch = chat_prompts[s:s + batch_size]
        inps = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inps,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, inps["input_ids"].shape[1]:]
        outs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        print(f"Generated {len(outs)}/{len(chat_prompts)}")

    return [o.strip() for o in outs]

def create_dataset(flag, 
                   df, 
                   path_out, 
                   window_width,
                   model,
                   tokenizer,
                   batch_size,
                   max_new_tokens,
                   function_words_file):
    if flag == "persuade":
        out_df = pd.DataFrame(columns=['task_prompt', 'output_text', 'WAN', 'label'])

        prompts = df["assignment"].tolist()
        human_text = df['full_text'].tolist()

        machine_text =  infer_MGT_instance(prompts = prompts,
                                           model =  model,
                                           tokenizer = tokenizer)

        out_df["task_prompt"] = prompts * 2
        out_df["output_text"] = list(human_text) + list(machine_text) 
        out_df["label"] = [0] * len(prompts) + [1] * len(prompts)
        out_df["WAN"] = [json.dumps(construct_wan_from_text(t, functionWordsFile=function_words_file, windowWidth=window_width)) for t in out_df["output_text"]]

        out_df.to_csv(path_out, index = False)
        print(f"Saved to {path_out}")

    else:
        raise NotImplementedError("only support for the Persuade dataset as of now")

    return 

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name)

    function_words_file = "data/function_words.txt"
    persuade_file = 'data/persuade_2.0_human_scores_demo_id_github.csv'


    df = pd.read_csv(persuade_file)
    df = df.sample(n=20)

    create_dataset(flag = 'persuade',
                   df = df,
                   model = model,
                   tokenizer = tokenizer,
                   path_out = "data/persuade_w_wan.csv",
                   window_width = 5,
                   batch_size = 4,
                   max_new_tokens = 700,
                   function_words_file= function_words_file)