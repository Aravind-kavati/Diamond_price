import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
import sys,os
from sklearn.preprocessing import OrdinalEncoder
from src.utils import save_object

from dataclasses import dataclass
@dataclass
class Datatransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
class Datatransformation:
    def __init__(self):
        self.data_transformation_config=Datatransformationconfig()


    def get_data_transforamtion_object(self):
        try:
            logging.info("data transformation initiated")
            ## catogorical and numerical varaibles

            
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']

            ## data making for label encoder 
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            logging.info("Pipeline initiated")
            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]
                  )
              # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]

            )
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            logging.info("exception occured in data transformation")
            raise CustomException(e,sys)
        
    def data_initiate_transforamtion(self,train_data_path,test_data_path):
        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)
            
            preprocessing_obj=self.get_data_transforamtion_object()

            target_column='price'
            drop_column=['id',target_column]

            input_feature_train_df=train_df.drop(drop_column,axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(drop_column,axis=1)
            targt_feature_test_df=test_df[target_column]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Data preprocessing completed")

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]

            test_arr=np.c_[input_feature_test_arr,np.array(targt_feature_test_df)]


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.info("Exception occured")
           
