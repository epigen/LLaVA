import argparse
import json
import os

import openai
import tqdm
import time

client = openai.OpenAI(api_key="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")  # Initialize the OpenAI client

def get_eval(system_prompt, content: str, max_tokens: int):
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(e)


    # return response['choices'][0]['message']['content']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    # parser.add_argument('-a', '--answer')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()


    f_q = open(os.path.expanduser(args.question))
    f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    f_ans2 = open(os.path.expanduser(args.answer_list[1]))
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    review_file = open(f'{args.output}', 'w')

    idx = 0
    for ques_js, ans1_js, ans2_js in tqdm.tqdm(zip(f_q, f_ans1, f_ans2)):
        # if idx == 1:
        #     break

        ques = json.loads(ques_js)
        ans1 = json.loads(ans1_js)
        ans2 = json.loads(ans2_js)

        category = json.loads(ques_js).get('category')
        if category in rule_dict:
            rule = rule_dict[category]
        else:
            rule = rule_dict['default']
        system_prompt = rule['prompt']
        role = rule['role']
        content = (f'[Question]\n{ques["text"]}\n\n[End of Question]\n\n'
                   f'[Reference]\n{ques["reference"]}\n\n[End of Reference]\n\n'
                   f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n')
        idx += 1
        res = get_eval(system_prompt, content, args.max_tokens)

        review_dict = json.loads(res)

        write_dict = ({
            'id': idx+1,
            'question_id': ques['question_id'],
            'answer1_id': ans1['answer_id'],
            'answer2_id': ans2['answer_id'],
            'content': review_dict["explanation"],
            'tuple': (review_dict['score_assistant1'], review_dict['score_assistant2']),
            'category': category})
        review_file.write(json.dumps(write_dict) + '\n')
    review_file.close()
