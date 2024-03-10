# Access Control Plan for FloorPlan 
## Process
1. Use [DataFloorPlans500](https://universe.roboflow.com/university-y9nbi/floor-plans-500/dataset/3) to train the first object recognition model. Since the data is not fully suitable on our usecase. This model is just for labeling.

    - Table Recognized as Door:![](static/figures_test_1st/Cat35_7.png)
    - Sink Recognized as Door:![](static/figures_test_1st/Cat47_5.png)
    - Door not Recognized:![](static/figures_test_1st/Cat49_1.png)
    
2. Use frist objection model to do the labeling with manual adjustment and train the second model based on our data format: [ROBIN](https://github.com/gesstalt/ROBIN). The manual adjusted data is at [google drive](https://drive.google.com/file/d/1hs6Qc2UjLSYC5Q9rDbENUcZDAgnrkf5y/view?usp=sharing), which can be put on the `FloorPlan/ROBIN/ROBIN_Door_15_Labeled`.

    - (Fixed) Table Recognized as Door:![](static/figures_test_2nd/Cat35_7.png)
    - (Fixed) Sink Recognized as Door:![](static/figures_test_2nd/Cat47_5.png)
    - (Fixed) Door not Recognized:![](static/figures_test_2nd/Cat49_1.png)


3. Use Traditional CV method to help Annotation the location of the door
    - Normal: 
        - ![](static/figures_orientationcv/Cat2_1_00_hc_detect.png)
        - ![](static/figures_orientationcv/Cat2_1_00_wall_loc.png)       
    - Fail1:
        - ![](static/figures_orientationcv/Cat15_2_00_hc_detect.png)
        - ![](static/figures_orientationcv/Cat15_2_00_wall_loc.png)       
    - Fail2:
        - ![](static/figures_orientationcv/Cat37_1_01_hc_detect.png)
        - ![](static/figures_orientationcv/Cat37_1_01_wall_loc.png)       
    - Fail3:
        - ![](static/figures_orientationcv/Cat39_9_03_hc_detect.png)
    - Fail4:
        - ![](static/figures_orientationcv/Cat30_4_00_hc_detect.png)       
        
        
4. Train the segmentation model to recognize the orientation of the door. The labeled dataset can be found at [google drive](https://drive.google.com/file/d/19rZ_CDH-WnGSQ_RE2iyY9PHb90tbEU1N/view?usp=sharing). The training code can be found in `FloorPlan/DoorOrientationSegmentation`
    - Labeled Data: ![](static/Segmentation/DatasetPreviewDsiffSample.png)   
    - Augmentation: ![](static/Segmentation/TransformFunctionsPreview_no_mask.png)
    - Model Prediction: ![](static/Segmentation/Evaluation.png)
    
5. Local Camera and Card Reader Location.
    - Prediction1: ![](static/final_predictions/Cat20_3.jpg)
    - Prediction2: ![](static/final_predictions/Cat21_2.jpg)

    
