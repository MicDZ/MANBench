import json

def eval_results(results_path):
    """
    Evaluate the results from the model and calculate the accuracy.

    Parameters:
    - results_path: String, the path to the results file.

    Returns:
    - acc, the accuracy of the model.
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    acc = {}
    for result in results:
        if result['category'] not in acc:
            acc[result['category']] = [0, 0]
        if result['prediction'] == result['answer']:
            acc[result['category']][0] += 1
        acc[result['category']][1] += 1
    acc_result = {}
    for category, (correct, total) in acc.items():
        print(f"Category: {category}, Accuracy: {(correct/total):.4f}")
        acc_result[category] = (correct/total) * 100

    return acc_result



            