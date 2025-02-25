import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.impute import SimpleImputer
import tensorflow_probability as tfp
import tensorflow.keras.backend as K
tf.keras.backend.set_floatx('float64')


def FeatureArrange(df):
    """
    This only for this project
    """
    # Rearrange the dataset
    ## numerical (17)
    num = ['BT_NM', 'HR_NM', 'RR_NM', 'HB_NM', 'HCT_NM', 'PLATELET_NM', 'WBC_NM', 'PTT1_NM',
           'PTT2_NM', 'PTINR_NM', 'ER_NM', 'BUN_NM', 'CRE_NM', 'BMI', 'age', 'NIHTotal',
           'PPD', ]
    ## category (55) + NIHSS (15)
    cat_NIHSS = ['THD_ID', 'THDA_FL', 'THDH_FL', 'THDI_FL',
           'THDAM_FL', 'THDV_FL', 'THDE_FL', 'THDM_FL', 'THDR_FL', 'THDP_FL',
           'THDOO_FL', 'Gender', 'cortical_ACA_ctr', 'cortical_MCA_ctr', 'subcortical_ACA_ctr',
           'subcortical_MCA_ctr', 'PCA_cortex_ctr', 'thalamus_ctr',
           'brainstem_ctr', 'cerebellum_ctr', 'Watershed_ctr',
           'Hemorrhagic_infarct_ctr', 'cortical_ACA_ctl', 'cortical_MCA_ctl',
           'subcortical_ACA_ctl', 'subcortical_MCA_ctl', 'PCA_cortex_ctl',
           'thalamus_ctl', 'brainstem_ctl', 'cerebellum_ctl', 'Watershed_ctl',
           'Hemorrhagic_infarct_ctl', 'cortical_CT', 'subcortical_CT',
           'circulation_CT',  'watershed_CT', 'Hemorrhagic_infarct_CT',
           'CT_left', 'CT_right', 'CT_find',
           'NIHS_1a_in', 'NIHS_1b_in', 'NIHS_1c_in',
           'NIHS_2_in', 'NIHS_3_in', 'NIHS_4_in', 'NIHS_5aL_in', 'NIHS_5bR_in',
           'NIHS_6aL_in', 'NIHS_6bR_in', 'NIHS_7_in', 'NIHS_8_in', 'NIHS_9_in',
           'NIHS_10_in', 'NIHS_11_in',
                ]

    ## Label (1)
    label = df[['elapsed_class']]
    # imputation
    imp_mode = SimpleImputer(strategy='most_frequent')
    imp_mean = SimpleImputer(strategy='mean')
    df[cat_NIHSS] = imp_mode.fit_transform(df[cat_NIHSS])
    df[num] = imp_mean.fit_transform(df[num])

    df = pd.concat([df[num], df[cat_NIHSS], label], axis=1)

    return df, imp_mode, imp_mean


def DataArrange2D(df, dim):
    '''
    transform to N*N matrix (fill with NaN) and MinMaxScaler this matrix
    input ->  dataframe and dimension
    output -> dataframe images and MinMax Scaler
    '''
    df_fa, imp_mode, imp_mean = FeatureArrange(df)
    sc = MinMaxScaler()
    df = sc.fit_transform(df_fa)

    # transform to N*N matrix (fill with NaN)
    df_img = []
    for i in range(len(df)):
        df_img.append(
            np.pad(df[i], (0, dim*dim-len(df)), constant_values=np.nan).reshape(dim, dim))
    df_img = np.array(df_img)

    return df, sc, imp_mode, imp_mean


def random_weight_average(x, x_gen):
    epsilon = tf.random.uniform(
        [x.shape[0], 1, ], 0, 1, dtype=tf.dtypes.float64)

    return epsilon*x+(1-epsilon)*x_gen


def discriminator_loss(real_output, gen_output, d_hat, x_hat, lambda_=10):
    real_loss = tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(gen_output)
    gp_loss = gradient_penalty(d_hat, x_hat)

    return fake_loss - real_loss + gp_loss*lambda_


def gradient_penalty(d_hat, x_hat):
    gradients = tf.gradients(d_hat, x_hat)
    # calculate L2 norm
#     gradients_sqr = tf.square(gradients)
#     gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=1, keepdims=True)
#     gradients_l2_norm = tf.sqrt(gradients_sqr_sum)
    
    ## the easiest way to calcuate L2 norm
    gradients_l2_norm = tf.norm(gradients, keepdims=True)
    gp = tf.reduce_mean(tf.square((gradients_l2_norm-1.)))

    return gp


def generator_loss(gen_output):
    return -tf.reduce_mean(gen_output)


def weight_L2(w, a, b):
    '''
    w -> weight\n
    a -> original data\n
    b -> generate data
    '''
    q = a-b
    return tf.sqrt(tf.reduce_sum(w*q*q))


def tf_entropy(inputs):
    _, _, count = tf.unique_with_counts(inputs)
    prob = count/tf.reduce_sum(count)
    return -tf.reduce_sum(prob*tf.math.log(prob))


def identifiability(gen_output, orig_data, min_entr=1e-3):
    iden_loss = tf.cast(0., dtype=tf.float64)
    ## Set the minimum of entropy
    min_entr = tf.cast(min_entr, dtype=tf.float64)
    biochemistry = [
        'HCT_NM', 'PLATELET_NM', 'WBC_NM', 'PTT1_NM', 'PTT2_NM',
        'PTINR_NM', 'ER_NM', 'BUN_NM', 'CRE_NM',
    ]
    num = [
        'BT_NM', 'HR_NM', 'RR_NM', 'HB_NM',  'BMI', 'age', 'PPD',
    ]
    # cat = [
    #     'THDA_FL', 'THDH_FL', 'THDI_FL', 'THDAM_FL', 'THDV_FL',
    #     'THDE_FL', 'THDM_FL', 'THDR_FL', 'THDP_FL', 'THDOO_FL', 'Gender',
    #     'cortical_ACA_ctr', 'cortical_MCA_ctr', 'subcortical_ACA_ctr',
    #     'subcortical_MCA_ctr', 'PCA_cortex_ctr', 'thalamus_ctr',
    #     'brainstem_ctr', 'cerebellum_ctr', 'Watershed_ctr',
    #     'Hemorrhagic_infarct_ctr', 'cortical_ACA_ctl', 'cortical_MCA_ctl',
    #     'subcortical_ACA_ctl', 'subcortical_MCA_ctl', 'PCA_cortex_ctl',
    #     'thalamus_ctl', 'brainstem_ctl', 'cerebellum_ctl', 'Watershed_ctl',
    #     'Hemorrhagic_infarct_ctl', 
    # ]
    NIH_all = [
        'NIHS_1a_in', 'NIHS_1b_in', 'NIHS_1c_in',
        'NIHS_2_in', 'NIHS_3_in', 'NIHS_4_in', 'NIHS_5aL_in', 'NIHS_5bR_in',
        'NIHS_6aL_in', 'NIHS_6bR_in', 'NIHS_7_in', 'NIHS_8_in', 'NIHS_9_in',
        'NIHS_10_in', 'NIHS_11_in',
    ]
    
    ## we temporary don't calculate numerical data and NIHSS
    start = len(biochemistry)+len(num)
    end = len(NIH_all)
    
    for i in range(orig_data[:, start:(-end)].shape[1]):
        # calculate discrete entropy and inverse the values
        ## start+i -> ignore the numerical and NIHSS columns
        entr = tf_entropy(orig_data[:, start+i]) 
        
        # To avoid entropy is 0 -> infinity
        if tf.math.is_inf(1/entr) == True:
            entr = min_entr
        weight = 1/entr
        # output weight L2
        iden_loss += weight_L2(weight, orig_data[:, start+i], gen_output[:, start+i]) 

    return iden_loss


def reality_constraint(data, params):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # inverse data to original values
    dataset = tf_inverse_MinMaxScalar(data, params)
    
    # Condition: NIHSS total <=42 & >=1
    ## The last column is label, so we select the range from (-16~-1)
    NIHSS_sub_sum = tf.reduce_sum(dataset[:, -16:-1], 0)

    # if sum > 42 or <1 return False (0)
    NIHSS_result = tf.cast(tf.where((NIHSS_sub_sum <= 39) & (
        NIHSS_sub_sum >= 1), 1, 0), dtype=tf.float64)

    NIHSS_loss = cross_entropy(tf.ones_like(
        NIHSS_result, dtype=tf.float64), NIHSS_result)
    
    #################### constraint of NIHSS details ###################################
    ## The last column is label, so we select the range from (-16~-1)
    NIHSS_dataset = tf.round(dataset[:, -16:-1])
    # NIHSS 1a == 3 -> NIHSS XX == X
    condition = np.array(
        [None, 2, 2, None, None, 3, 4, 4, 4, 4, 0, 2, 3, 2, 2])

    for i, cond in enumerate(condition):
        if cond:
            # calculate the NIHSS detail's loss
            NIHSS_result = tf.cast(tf.where((NIHSS_dataset[:, 0] == 3) & (
                NIHSS_dataset[:, i] == cond), 1, 0), dtype=tf.float64)

            loss = cross_entropy(tf.ones_like(NIHSS_result, dtype=tf.float64),
                                 NIHSS_result, )
            # sum all the loss of NIHSS
            NIHSS_loss += loss 
    
    return NIHSS_loss


def tf_inverse_MinMaxScalar(data, params):
    # we can't convert tensor to numpy while we are training, so operate the inverse by tf api
    data = data * \
        tf.convert_to_tensor(
            (params['max']-params['min']))+tf.convert_to_tensor(params['min'])

    return data

def Mahalanobis_dist(m, n):
    diff = m - n
    tf_stack=tf.concat([x, y], axis=0)
    ## Calculate covariance
    try :
        V = tfp.stats.covariance(tf_stack)
    except Exception as ex:
        V = tf_covariance(tf_stack)
        
    VI = tf.linalg.inv(V)
    dist=tf.sqrt(K.dot(K.dot(diff,VI),tf.transpose(diff)))
    
    return tf.linalg.diag_part(dist)

## If your tensorflow version <= 2.4 , you need to use this to calculate covariance
def tf_covariance(x):
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    m = tf.matmul(tf.transpose(mean_x), mean_x)
    v = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float64)
    cov = v - m
    return  cov
    
