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
            self.L_path_model = "/opt/algorithm/checkpoints/L_MLP.pth"
            self.R_path_model = "/opt/algorithm/checkpoints/R_MLP.pth"
        else:
            self.L_path_model = "./weights/L_MLP.pth"
            self.R_path_model = "./weights/R_MLP.pth"

        #You may adapt this to your model/algorithm here.
        self.L_model = MLP(4, [16, 16, 16, 16], 1, device=self.device)
        self.R_model = MLP(4, [16, 16, 16, 16], 1, device=self.device)
        #loading model weights
        self.L_model.load_state_dict(torch.load(self.L_path_model, map_location=self.device),strict=False)
        self.R_model.load_state_dict(torch.load(self.R_path_model, map_location=self.device),strict=False)
        
        self.num = 0
    
    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)

    def process_case(self, *, idx, case):

        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)
        print(input_image_file_path)
        # Detect and score candidates
        prediction = self.predict(input_image=input_image, input_image_file_path=input_image_file_path)
        # Return a float for prediction
        return float(prediction)

    def predict(self, *, input_image: SimpleITK.Image, input_image_file_path: str) -> Dict:

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
            
        if execute_in_docker:
            means = torch.from_numpy(np.load('/opt/algorithm/utils/means.npy'))
            stds = torch.from_numpy(np.load('/opt/algorithm/utils/stds.npy'))
        else:
            means = torch.from_numpy(np.load('./utils/means.npy'))
            stds = torch.from_numpy(np.load('./utils/stds.npy'))

        image_data = (image_data - means.reshape(4, 1)) / stds.reshape(4, 1)

        image_sequence = Data(x=image_data)

        with torch.no_grad():
        
            if self.num % 2 == 0:
                prediction = self.L_model(image_sequence)
            else:
                prediction = self.R_model(image_sequence)
            self.num += 1
        
        return prediction.cpu().numpy()[0][0]

if __name__ == "__main__":

    Slcn_algorithm().process()

    
