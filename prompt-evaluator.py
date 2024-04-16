import json
import statistics
import time
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase

mistral_api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-small-latest"

faithfulness = FaithfulnessMetric(
    threshold=0.7,
    model="gpt-3.5-turbo",
    include_reason=True
)

relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-3.5-turbo",
    include_reason=True
)

hallucination = HallucinationMetric(threshold=0.5)

data_file = json.load(open("llm_requests.json"))

def main():
    client = MistralClient(api_key=mistral_api_key)

    print("Sample keys: '{wrong}', '{request}', '{right}', '{intermediate}'")
    print("Sample request: 'As a lawyer describe how '{wrong}' is related to {request}'")
    print("Type 'exit' to stop.")

    while True:
        query = input("\nEnter text: ")
        data_file['prompt'] = query
        if query == "exit":
            break
        if query.strip() == "":
            continue

        for data in data_file['data']:
            request = query.format(**data)

            messages = [
                ChatMessage(role="user", content=request)
            ]

            response = client.chat(
                model=model,
                messages=messages,
            )

            output = response.choices[0].message.content

            test_case = LLMTestCase(
                input=request,
                actual_output=output,
                retrieval_context=list(data['right']),
                context=list(data['right'])
            )

            print(output)
            data['output'] = output

            faithfulness.measure(test_case)
            data['faithfulness'] = faithfulness.score
            data['faithfulness_reason'] = faithfulness.reason
            print(faithfulness.score)
            print(faithfulness.reason)

            relevancy.measure(test_case)
            data['relevancy'] = relevancy.score
            data['relevancy_reason'] = relevancy.reason
            print(relevancy.score)
            print(relevancy.reason)

            hallucination.measure(test_case)
            data['hallucination'] = hallucination.score
            print(hallucination.score)

            print('\n\n\n')

        faith = [o['faithfulness'] for o in data_file['data']]
        faith_avg = sum(faith) / len(faith)
        faith.append(statistics.median(faith))
        faith.append(faith_avg)

        rel = [o['relevancy'] for o in data_file['data']]
        rel_avg = sum(rel) / len(rel)
        rel.append(statistics.median(rel))
        rel.append(rel_avg)

        hal = [o['hallucination'] for o in data_file['data']]
        hal_avg = sum(hal) / len(hal)
        hal.append(statistics.median(hal))
        hal.append(hal_avg)

        print("Faith\t" + "\t".join(format(x, "10.3f") for x in faith) + "\t" + query)
        print("Rel\t" + "\t".join(format(x, "10.3f") for x in rel))
        print("Hal\t" + "\t".join(format(x, "10.3f") for x in hal))

        filename = f'experiment_{round(time.time() * 1000)}.json'
        with open(filename, 'w') as f:
            json.dump(data_file, f)
            print(f"result saved to '{filename}'")



if __name__ == '__main__':
    main()