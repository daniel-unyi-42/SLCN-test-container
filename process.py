# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2022-04-20 10:10:19
# @Last Modified by:   Your name
# @Last Modified time: 2022-05-04 10:38:36

from typing import Dict

import SimpleITK
import numpy as np

from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

#### Import librairies requiered for your model and predictions
from torch_geometric.data import Data
import torch
from model.mlp import MLP
import pandas as pd
from pathlib import Path
import json
from glob import glob

execute_in_docker = True

class Slcn_algorithm(ClassificationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path = Path("/input/images/cortical-surface-mesh/") if execute_in_docker else Path("./test/"),
            output_file= Path("/output/birth-age.json") if execute_in_docker else Path("./output/birth-age.json")
        )
        
        ###                                                                                                     ###
        ###  TODO: adapt the following part for YOUR submission: should create your model and load the weights
        ###                                                                                                     ###

        # use GPU if available otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("===> Using ", self.device)

        #This path should lead to your model weights
        if execute_in_docker:
            self.path_model = "/opt/algorithm/checkpoints/ckpt.pth"
        else:
            self.path_model = "./weights/ckpt.pth"

        #You may adapt this to your model/algorithm here.
        self.model = MLP(4, [16, 16, 16, 16], 1, device=self.device)
        #loading model weights
        self.model.load_state_dict(torch.load(self.path_model,map_location=self.device),strict=False)
    
    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, _ = self._load_input_image(case=case)
        # Detect and score candidates
        prediction = self.predict(input_image=input_image)
        # Return a float for prediction
        return float(prediction)

    def predict(self, *, input_image: SimpleITK.Image) -> Dict:

        # Extract a numpy array with image data from the SimpleITK Image
        image_data = SimpleITK.GetArrayFromImage(input_image)

        ###                                                                     ###
        ###  TODO: adapt the following part for YOUR submission: make prediction
        ###                                                                     ###

        ## input image of shape (N vertices, C channels)
        if image_data.shape[0]==4:
            pass
        else:
            image_data = np.transpose(image_data, (1,0))

        image_sequence = Data(x=image_data, batch=1)

        with torch.no_grad():

            prediction = self.model(image_sequence)
        
        return prediction.cpu().numpy()[0][0]

if __name__ == "__main__":

    Slcn_algorithm().process()

    
