from typing import Dict, List, Any

import requests

from scipy.sparse import coo_matrix


class ConceptNetClient:
    """
    This class is used to query the ConceptNet API.
    """

    def __init__(self, host="http://localhost:8080"):
        self.host = host

    def resolve_csqa(self, common_sense_qa_example: Dict[str, str]):
        response = None
        r = requests.get(url=self.host + "/csqa",
                         params=common_sense_qa_example,
                         timeout=5)
        if r.status_code == 200:
            response = self.json_to_subgraph(r.json())

        return r.status_code, response

    def json_to_subgraph(self, json_response: List[Dict[str, str]]):
        response = []
        for csqa_example in json_response:
            response.append(self.parse_response_csqa_example(csqa_example))
        return response

    def parse_response_csqa_example(self, csqa_response: Dict[str, Any]):
        return {"adj": self.adj_list_to_coo_matrix(csqa_response["adj"]),
                "concepts": csqa_response["concepts"],
                "qmask": csqa_response["qmask"],
                "amask": csqa_response["amask"],
                "cid2score": [(i[0], i[1]) for i in csqa_response["cid2score"]],
                }

    def adj_list_to_coo_matrix(self, adj_list: List[List[int]]):
        """
        It takes a list of lists, where the first list is the row indices and the second list is the column indices, and
        returns a sparse matrix in COOrdinate format

        :param adj_list: A list of two lists, the first list is the row indices, the second list is the column indices
        :type adj_list: List[List[int]]
        :return: A sparse matrix in COOrdinate format.
        """
        assert len(adj_list) == 2  # rows, cols
        rows = adj_list[0]
        cols = adj_list[1]
        assert len(rows) == len(cols)
        data = [1] * len(rows)
        return coo_matrix((data, (rows, cols)), shape=(len(rows), len(cols)))
