import tritonclient.grpc as grpcclient
import numpy as np
import cv2
import sys
from numpy.linalg import norm

def l2_norm(input, axis = 1):
    normed = norm(input)
    output = input/normed
    return output

def d_process(im):
    im = np.transpose(im, (2, 0, 1))
    im = (im / 255. - 0.5) / 0.5
    return im

def preprocess(image, input_width, input_height):
    image = np.float32(image)
    
    # img = cv2.resize(image, (input_width, input_height))
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    img_flip = cv2.flip(img, 1)
    
    im1 = d_process(img)        
    
    return im1

def preprocess_batch(batch_image, input_width, input_height):
    batch_image = np.array([preprocess(image, input_width, input_height) for image in batch_image]) 
    print(batch_image.shape)
    return batch_image

class TritonDetector():
    def __init__(self,
            model = 'face_detection',
            input_width = 640,
            input_height = 640,
            mode = "FP32",
            url = '10.70.110.251:8001',
            verbose = False,
            ssl = None,
            root_certificates = None,
            private_key = None,
            certificate_chain = None,
            client_timeout = None,
            batch_size = 1):
            
        self.model = model
        self.input_width = input_width
        self.input_height = input_height
        self.mode = mode
        self.client_timeout = client_timeout
        self.batch_size = batch_size
        
        self.inputs = []
        self.outputs = []
        # Create server context
        try:
            self.triton_client = grpcclient.InferenceServerClient(
                url=url,
                verbose=verbose,
                ssl=ssl,
                root_certificates=root_certificates,
                private_key=private_key,
                certificate_chain=certificate_chain)
        except Exception as e:
            print("context creation failed: " + str(e))
            sys.exit()

        # Health check
        if not self.triton_client.is_server_live():
            print("FAILED : is_server_live")
            sys.exit(1)

        if not self.triton_client.is_server_ready():
            print("FAILED : is_server_ready")
            sys.exit(1)
        
        if not self.triton_client.is_model_ready(model):
            print("FAILED : is_model_ready")
            sys.exit(1)

    def detect(self, image):
        self.inputs.append(grpcclient.InferInput('input0', [self.batch_size, 3, self.input_height, self.input_width], "FP32"))
        self.outputs.append(grpcclient.InferRequestedOutput('output'))
        self.inputs[0].set_data_from_numpy(image)

        results = self.triton_client.infer(model_name=self.model,
                                    inputs=self.inputs,
                                    outputs=self.outputs,
                                    client_timeout=self.client_timeout)
        print(len(results.as_numpy('output')))

if __name__ == '__main__':
    import time
    import numpy as np
    extractor = TritonDetector()

    im = np.zeros((1,3,640,640),dtype=np.float32)
    # extractor.detect(im)
