from ..base_model.base_model import BaseModel


class FaceReidentifier(BaseModel):
    def __init__(self, device=None,
                 model_fp=None, model_dir=None,
                 cpu_extension=None, path_config=None):
        self.task = 'face_reidentification'
        super().__init__(self.task, device,
                         model_fp, model_dir,
                         cpu_extension, path_config)
    
    def pre_process(self, input_frame):
        return input_frame
    
    def post_process(self, results):
        # TODO: is it correct?
        outputs = results['658']
        outputs = outputs.flatten()
        return outputs
    
    def compute(self, input_frame):
        """calculate from person image to vector  
        
        Args:
            frame (np.ndarray): input person cropped image
        
        Returns:
            np.ndarray: feature vector
        """
        
        processed_frame = self.pre_process(input_frame)
        results = self.get_result(processed_frame)
        outputs = self.post_process(results)
        return outputs
    
    
