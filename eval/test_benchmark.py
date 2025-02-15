import json
from tqdm import tqdm
import os
from multiple_choice import match_multiple_choice
import argparse
from datasets import load_dataset, load_from_disk
from query_model import query_OpenAI
from huggingface_hub import login
from cal_result import eval_results

def load_prompt(d, lang, force_choose=False):
    """
    Loads the prompt and images from huggingface data entry, and formats the prompt in the correct language.

    Parameters:
    - task_name: String, the name of the task.
    - lang: String, the language, either 'english' or 'chinese'.
    - d: Data entry, the data dictionary containing the prompt and images.
    - force_choose: Boolean, whether to force the model to choose an answer.


    Returns:
    - imags: List of strings, the images.
    - prompt: String, the prompt text.
    """
    if lang == "english":
        question_prompt =json.loads(d['question']).get('english')

        force_choose_prompt = "Even if the images is not provided or you are not sure about the answer, you are forced to choose one of the options."
        select_prompt = "Please select the correct answer from the following options:"
        options_prompt = ""
        for i, option in enumerate(json.loads(d['options'])):
            options_prompt += f"({chr(65 + i)}) {option.get('english')}\n"
    elif lang == "chinese":
        question_prompt = json.loads(d['question']).get('chinese')
        force_choose_prompt = "即使图片未提供或者你不确定答案，你也需要强制选择一个选项。"
        select_prompt = "请从以下选项中选择正确答案："
        options_prompt = ""
        for i, option in enumerate(json.loads(d['options'])):
            options_prompt += f"({chr(65 + i)}) {option.get('chinese')}\n"
    else:
        raise ValueError("Invalid language")
    

    prompt = ""
    prompt += question_prompt
    prompt += "\n"
    if force_choose:
        prompt += force_choose_prompt
        prompt += "\n"
    prompt += select_prompt
    prompt += "\n"
    prompt += options_prompt
    prompt += "\n"

    print(prompt)
    return prompt

def analyze_answer(gpt_answer, prompt):
    """
    extracts the multiple choice answer from a long paragraph of model output if there is only one choice; otherwise, query GPT3.5 turbo to extract the choice. If the model output is short and only contains the choice, reformats the choice in the correct format e.g. (A) and returns the choice as is.

    Parameters:
    - gpt_answer: String, the model output.
    - all_choices: List of strings, the list of all choices.

    Returns:
    - prediction, the extracted answer.
    """
    try:
        if gpt_answer in ["A", "B", "C", "D"]:
            prediction = "(" + gpt_answer + ")"
        elif gpt_answer in ['(A)', '(B)', '(C)', '(D)']:
            prediction = gpt_answer
        else:
            extracted_answer = match_multiple_choice(prompt, gpt_answer)
            prediction = extracted_answer
        return prediction
    except Exception as e:
        pass
        # print(i, e)


def eval_tasks(categories, dataset, model_generate_func, lang, model_name, id, mask_image=False):
    """
    Evaluate the tasks from the dataset and calculate the accuracy.

    Parameters:
    - categories: List of strings, the categories to evaluate.
    - dataset: dataset, the dataset to evaluate.
    - model_generate_func: function, the function to query the model.
    - lang: String, the language.

    Returns:
    - results, the results of the model.
    - acc, the accuracy of the model.
    """
    result_path = f'eval/saved_mllms_results/{lang}/{model_name}_{lang}_{"maskimage_" if mask_image else ""}results.json'

    results = []

    print(f'Categories: {categories}')
    dataset_filter = dataset.filter(lambda x: x['category'] in categories)
    print(f'Number of eval data: {len(dataset_filter)}')
    acc = {}
    # results = []
    for i, data in enumerate(tqdm(dataset_filter)):
        if id != -1 and data['idx'] != id:
            continue
        item = None
        for result in results:
            if result['idx'] == data['idx']:
                item = result
                break
        if item is not None and item['prediction']!='(Z)' and id == -1:
            continue

        print(f'Index: {i}')

        images_path = []
        if not os.path.exists('saved_images'):
            os.makedirs('saved_images')
        if not mask_image:
            for k in ['image_0', 'image_1', 'image_2', 'image_3', 'image_4']:
                if k in data and data[k]:
                    image = data[k]
                    image_path = f'saved_images/{data["idx"]}_{k[-1]}.jpg'
                    image.save(image_path)
                    images_path.append(image_path)
                
        prompt = load_prompt(data, lang, mask_image)
        output = model_generate_func(images_path, prompt)
        prediction = analyze_answer(output, prompt)
        answer = data['answer']
        result = {
            'idx': data['idx'],
            'answer': f'({answer})',
            'full_output': output,
            'prediction': prediction,
            'category': data['category'],
        }
        if acc.get(data['category']) is None:
            acc[data['category']] = [0, 0]
        if prediction == f'({answer})':
            acc[data['category']][0] += 1
        acc[data['category']][1] += 1
        print(f'Prediction: {prediction}')
        if item is None:
            results.append(result)
        else:
            item['prediction'] = prediction
            item['full_output'] = output
        with open(result_path, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    return results, acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='GPT-4o', help="Select the model.")
    parser.add_argument("--category", type=str, default='all', help="Select the task.")
    parser.add_argument("--lang", type=str, default='english', help="Select the language, either 'english' or 'chinese'.")

    parser.add_argument("--id", type=int, default=-1, help="Select the specific question.")
    parser.add_argument("--mask_image", type=bool, default=False, help="Whether to mask the image")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_name = args.model_name
    lang = args.lang
    id = args.id
    mask_image = args.mask_image
    print(f'Using model: {model_name}, language: {lang}, id: {id}, mask_image: {mask_image}')

    if args.category == 'all':
        categories = ['Text-image Understanding', 'Transmorphic Understanding', 'Multi-image Understanding', 'Numbers Pattern', 'Knowledge', 'Puzzles', 'Image Consistency', 'Text Locating', 'Spatial Imagination']
    else:
        categories = [args.category]

    # Feel free to add more models here
    model_generate_funcs = {'GPT-4o': query_OpenAI}

    model_generate_func = model_generate_funcs[model_name]
    
    dataset_name = 'MANBench/MANBench'
    dataset = load_dataset(dataset_name, split='train')
   
    results, acc = eval_tasks(categories, dataset, model_generate_func, lang, model_name, id, mask_image)

    if not os.path.exists('eval/saved_mllms_results'):
        os.makedirs('eval/saved_mllms_results')
    if not os.path.exists(f'eval/saved_mllms_results/{lang}'):
        os.makedirs(f'eval/saved_mllms_results/{lang}')

    result_path = f'eval/saved_mllms_results/{lang}/{model_name}_{lang}_{"maskimage_" if mask_image else ""}results.json'
    # Save the results
    with open(result_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    tasks_acc = eval_results(result_path)
    print(tasks_acc)

if __name__ == '__main__':
    main()
