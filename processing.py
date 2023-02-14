##### IMPORTS #####
import cv2


from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
##### IMPORTS #####\


##### METHOD: IMMEDIATE PROCESS #####
def immediate_process(img, predictor):
   im = cv2.imread(img)
   shape = im.shape
   im = cv2.resize(im, (640, 640))
   outputs = predictor(im)


   v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get("MetadataCatalog"), scale=1)
   out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
   return cv2.resize(out.get_image(), (shape[1], shape[0]))[:, :, ::-1]
##### METHOD: IMMEDIATE PROCESS #####\
