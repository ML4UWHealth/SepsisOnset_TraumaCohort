import os

class ProjectPaths:
    def __init__(self, base_path):
        self.base_path = base_path
        # data folder: raw data, preprocess data
        self.data_path = os.path.join(self.base_path, 'data')
        self.raw_data_path = os.path.join(self.data_path, 'raw')
        self.processed_data_path = os.path.join(self.data_path, 'processed')
        
        # dataset folder:
        self.final_dataset_path = os.path.join(self.base_path, 'dataset')
        # patient ids fold assignments for stratified 5-fold CV at the patient level.
        project_path_obj.fold_patient_info_path = os.path.join(self.final_dataset_path, 'Fold_IDs.csv')
        # dataset with nan value
        self.dataset_with_nan_path = os.path.join(self.final_dataset_path, 'PostTraumaticSepsis_dataset_w_nan.pkl')
        # dataset w/o nan value
        self.dataset_wo_nan_path = os.path.join(self.final_dataset_path, 'PostTraumaticSepsis_dataset_wo_nan.pkl')

        
        # code folder
        self.scripts_path = os.path.join(self.base_path, 'scripts')
        # self.notebooks_path = os.path.join(self.base_path, 'notebooks')
        self.src_path = os.path.join(self.base_path, 'src')
        # self.tests_path = os.path.join(self.base_path, 'tests')
        # documents folder
        self.supplementary_path = os.path.join(self.base_path, 'supplementary')
        self.docs_path = os.path.join(self.base_path, 'docs')




        # important files saved in "raw" folder

        # important files saved in "processed" folder
        self.trauma_cohort_info_path = os.path.join(self.processed_data_path, 'trauma_cohort_info.csv')
        # self.trauma_blood_cx_path = os.path.join(self.processed_data_path, 'trauma_blood_cx.csv')
        self.trauma_abxOrder_path = os.path.join(self.processed_data_path, 'trauma_abx_order.csv') # abx order 
        self.trauma_abxEvent_path = os.path.join(self.processed_data_path, 'trauma_abx_event.csv') # abx event 
        # self.trauma_sofa_path = os.path.join(self.processed_data_path, 'trauma_sofa.csv')
        self.sepsis_label_path = os.path.join(self.processed_data_path, 'sepsis_label.csv') # sepsis onset info

        


    # def get_file_path(self, *path_segments):
    #     return os.path.join(self.base_path, *path_segments)

    def get_raw_data_file(self, filename):
        return os.path.join(self.raw_data_path, filename)

    def get_processed_data_file(self, filename):
        return os.path.join(self.processed_data_path, filename)

    def get_final_data_file(self, filename):
        return os.path.join(self.final_data_path, filename)

    def get_script_file(self, filename):
        return os.path.join(self.scripts_path, filename)

    # def get_notebook_file(self, filename):
    #     return os.path.join(self.notebooks_path, filename)

    def get_src_file(self, filename):
        return os.path.join(self.src_path, filename)

    # def get_doc_file(self, filename):
    #     return os.path.join(self.docs_path, filename)

    # def get_test_file(self, filename):
    #     return os.path.join(self.tests_path, filename)

    def get_supplementary_file(self, filename):
        return os.path.join(self.supplementary_path, filename)

# # Example usage
# base_path = '/your/project/base/path'
# project_paths = ProjectPaths(base_path)

# # Access a specific file
# icd_file_path = project_paths.get_supplementary_file('ICD_Nonpoisoning_Cause_Matrix.xlsx')
# print(icd_file_path)
