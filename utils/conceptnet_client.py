from collections import OrderedDict
from typing import Dict, List, Any

import numpy as np
import requests

from scipy.sparse import coo_matrix


class ConceptNetClient:
    """
    This class is used to query the ConceptNet API.
    """

    def __init__(self, host="http://localhost:8080"):
        self.host = host

    def resolve_csqa(self, common_sense_qa_example: Dict[str, str]):
        response = {"result": None}
        r = requests.post(url=self.host + "/csqa/",
                          json=common_sense_qa_example)

        response["status_code"] = r.status_code
        if r.status_code == 200:
            response["result"] = self.json_to_subgraph(r.json())

        return response

    def json_to_subgraph(self, json_response: List[Dict[str, str]]):
        response = []
        for csqa_example in json_response:
            response.append(self.parse_response_csqa_example(csqa_example))
        return response

    def parse_response_csqa_example(self, csqa_response: Dict[str, Any]):
        return {"adj": self.adj_list_to_coo_matrix(csqa_response["adj"], csqa_response["adj_shape"]),
                "concepts": np.array(csqa_response["concepts"], dtype=np.int32),
                "qmask": np.array(csqa_response["qmask"]),
                "amask": np.array(csqa_response["amask"]),
                "cid2score": OrderedDict([(i[0], i[1]) for i in csqa_response["cid2score"]]),
                }

    def adj_list_to_coo_matrix(self, adj_list: List[List[int]], adj_shape: List[int]):
        """
        It takes in a list of lists, where the first list is the row indices and the second list is the column indices, and
        returns a sparse matrix with the same shape as the input shape

        :param adj_list: List[List[int]] rows and columns of the adjacency matrix
        :type adj_list: List[List[int]]
        :param adj_shape: The shape of the adjacency matrix
        :type adj_shape: List[int]
        :return: A sparse matrix in COOrdinate format.
        """
        assert len(adj_list) == 2  # rows, cols
        rows = adj_list[0]
        cols = adj_list[1]
        assert len(rows) == len(cols)
        data = np.array([1] * len(rows), dtype=np.uint8)
        return coo_matrix((data, (rows, cols)), shape=(adj_shape[0], adj_shape[1]))