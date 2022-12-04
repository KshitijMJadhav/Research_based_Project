from keras.utils import np_utils
from keras.layers import add, Conv2D,Input,BatchNormalization,TimeDistributed,Embedding,LSTM,GRU,Dense,MaxPooling1D,Dropout,LeakyReLU,ReLU,Flatten,concatenate,Bidirectional
from keras.layers import concatenate
from keras.models import Model,load_model

def InstantiateModel(in_):
  '''
    Architecture of the Deep Learning Model.
    Args:
      in_: input tensor shape
    Returns: Tensor model
  '''
  model_2_1 = GRU(32,return_sequences=True,activation=None,go_backwards=True)(in_)
  model_2 = LeakyReLU()(model_2_1)
  model_2 = GRU(128,return_sequences=True, activation=None,go_backwards=True)(model_2)
  #model_2 = BatchNormalization()(model_2)
  model_2 = LeakyReLU()(model_2)
  
  model_3 = GRU(64,return_sequences=True,activation=None,go_backwards=True)(in_)
  model_3 = LeakyReLU()(model_3)
  model_3 = GRU(128,return_sequences=True, activation=None,go_backwards=True)(model_3)
  #model_3 = BatchNormalization()(model_3)
  model_3 = LeakyReLU()(model_3)
 
  return model_3