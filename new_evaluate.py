import torch

from modeling.modeling_greaselm import GreaseLMForMultipleChoice
from utils.graph_loader import GraphLoader


def index_to_answer(index, choices=["A", "B", "C", "D", "E"]):
    return choices[index]


def evaluate():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    graph_loader = GraphLoader(url="http://localhost:8080",
                               batch_size=2,
                               device=(device, device),  # don't ask :-)
                               model_name="roberta-large")

    csqa = {"answerKey": "A", "id": "1afa02df02c908a558b4036e80242fac",
            "question": {"question_concept": "revolving door",
                         "choices": [{"label": "A", "text": "bank"}, {"label": "B", "text": "library"},
                                     {"label": "C", "text": "department store"}, {"label": "D", "text": "mall"},
                                     {"label": "E", "text": "new york"}],
                         "stem": "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?"}}

    qids, labels, single_batch = graph_loader.resolve_csqa(csqa)
    model = GreaseLMForMultipleChoice.from_pretrained("./greaselm_model/")
    model.resize_token_embeddings(len(graph_loader.tokenizer))

    model.to(device)
    model.eval()

    output: GreaseLMForMultipleChoice = model(**single_batch.to(device))
    result = output.logits.argmax(1) == labels.to(device)
    model_answer = index_to_answer(output.logits.argmax(1).item())
    correct_answer = index_to_answer(labels.item())
    if result.item():
        print(f"Model answered correctly, answer is {model_answer}")
    else:
        print(F"Model answered incorrectly, answer is not {model_answer} but {correct_answer}")


if __name__ == '__main__':
    evaluate()
