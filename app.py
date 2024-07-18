from fastapi import FastAPI, UploadFile, File, Query, Form
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
import requests
from PIL import Image
from io import BytesIO
import hashlib

import os
import torch
import mmcv
from mmcv import Config
from mmdet.apis import inference_detector, show_result_pyplot,init_detector

app = FastAPI()

# enable CORS (Cross-Origin Resource Sharing) to allow requests from any domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# define a route for uploading an image and processing it with a simplified version of token authorization
@app.post("/process_image/")
async def process_image(url: str = Form(...), token: str = Form(..., description="Authentication token")):
    if hashlib.sha256(str.encode(token)).hexdigest()=="7704e3e7fc812fe88942f2a1d01ef809a041cc18eabc09b164baa227fca52cad":
        try:
            # make an HTTP GET request to the image URL
            response = requests.get(url)

            # check if the request was successful
            response.raise_for_status()
            #-------------------------------------------------------

            # this part should be modified for future code extensions where gpu is available
            if torch.cuda.is_available():
                use_device = "cpu"
            else:
                use_device = "cpu"

            # set-up local paths and files
            # the code here is implemented for possible future extension to include more models
            architecture_selected = "mask_rcnn_r50_fpn_1x_coco"
            configs_folder = "configs/"

            # find the trained weights and the configuration of the model
            files = os.listdir(configs_folder)
            architecture_config_file = [x for x in files if x.endswith(".py")][0]
            architecture_pretrained_file = configs_folder + [x for x in files if x.endswith(".pth")][0]
            config_save_filename = "custom_config_" + architecture_selected
            
            # LOAD CONFIG FILE FROM CUSTOM SAVED .PY FILE AND CHANGE THE PRETRAINED WEIGHT PATH
            cfg = Config.fromfile(configs_folder + config_save_filename + ".py")
            cfg["load_from"] = configs_folder + architecture_selected + "_BEST_mAP.pth"

            # build the model from the config file and the checkpoint file that exists inside the config as a path
            model = init_detector(cfg, cfg["load_from"], device=use_device)

            # set the save path for the processed image
            processed_image_path = "images/processed_image.png"  

            # extract and save the image from the message
            image = Image.open(BytesIO(response.content))
            image.save("images/downloaded_image.png")

            # read the image in an acceptable format from MMDetection library
            img = mmcv.imread("images/downloaded_image.png")
            # run the model on the image
            result = inference_detector(model, img)
            # visualize the results, save visualization, score_thr is to determine the minimum confidence of a prediction in order to be visualized, by default use 0.5
            show_result_pyplot(model, img, result, out_file = processed_image_path,score_thr=0.5)
            
            return FileResponse(processed_image_path)

        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to download image from URL: {str(e)}"}
    else:
        return {"error": "Not authorized"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)