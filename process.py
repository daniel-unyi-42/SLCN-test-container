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
import torch
from model.mlp import MLP
import pandas as pd
from pathlib import Path
import json
from glob import glob
import nibabel as nib

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
        self.device = torch.device("cpu")
        print("===> Using ", self.device)
        
        
        # current model
        if execute_in_docker:
            self.path_model = "/opt/algorithm/checkpoints/MLP_template.pt"
            self.neigh_orders = np.load('/opt/algorithm/utils/neigh_orders.npy')
            self.mirror_index = np.load('/opt/algorithm/utils/mirror_index.npy')
            self.means = np.load('/opt/algorithm/utils/means_template.npy')
            self.stds = np.load('/opt/algorithm/utils/stds_template.npy')
            self.Lref = nib.load('/opt/algorithm/utils/Lref_template.gii')
        else:
            self.path_model = "./weights/MLP_template.pt"
            self.neigh_orders = np.load('./utils/neigh_orders.npy')
            self.mirror_index = np.load('./utils/mirror_index.npy')
            self.means = np.load('./utils/means_template.npy')
            self.stds = np.load('./utils/stds_template.npy')
            self.Lref = nib.load('./utils/Lref_template.gii')
        self.Lref = np.stack(self.Lref.agg_data(), axis=1)
        self.model = MLP(28, [28, 28, 28, 28], 3, device=self.device)
        self.model.load_state_dict(torch.load(self.path_model))
        self.model.eval()
        
        

#        #This path should lead to your model weights
#        if execute_in_docker:
#            self.L_path_model = "/opt/algorithm/checkpoints/LMLP.pt"
#            self.R_path_model = "/opt/algorithm/checkpoints/RMLP.pt"
#        else:
#            self.L_path_model = "./weights/LMLP.pt"
#            self.R_path_model = "./weights/RMLP.pt"
#
#        #You may adapt this to your model/algorithm here.
#        self.L_model = MLP(4, [16, 16, 16, 16], 1, device=self.device)
#        self.R_model = MLP(4, [16, 16, 16, 16], 1, device=self.device)
#        #loading model weights
#        self.L_model.load_state_dict(torch.load(self.L_path_model))
#        self.R_model.load_state_dict(torch.load(self.R_path_model))
#        self.L_model.eval()
#        self.R_model.eval()
    
    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)

    def process_case(self, *, idx, case):

        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)
        # Detect and score candidates
        prediction = self.predict(input_image=input_image, input_image_file_path=input_image_file_path)
        # Return a float for prediction
        return float(prediction)

    def predict(self, *, input_image: SimpleITK.Image, input_image_file_path: str) -> Dict:

        # Extract a numpy array with image data from the SimpleITK Image
        image_data = SimpleITK.GetArrayFromImage(input_image)
        
        print(image_data.shape)

        ###                                                                     ###
        ###  TODO: adapt the following part for YOUR submission: make prediction
        ###                                                                     ###

        ## input image of shape (N vertices, C channels)
        if image_data.shape[1]==4:
            pass
        else:
            image_data = np.transpose(image_data, (1,0))

        print(image_data)

        error = np.absolute(np.subtract(image_data, self.Lref)).mean()
        
        print(error)

        image_data = image_data[self.neigh_orders].reshape([image_data.shape[0], 28])
        
        print(image_data)
        
        print(self.mirror_index)
        
        if error > 1.0:
            print('Here')
            image_data = image_data[self.mirror_index]

        print(image_data)
        
        image_data = (image_data - self.means) / self.stds
        
        print(image_data)

        with torch.no_grad():
        
            prediction = self.model(torch.from_numpy(image_data)).cpu().numpy()

        print(prediction, prediction[0][0])

        return prediction[0][0]

if __name__ == "__main__":

    Slcn_algorithm().process()

    
