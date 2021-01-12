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


def FeatureArrange(df):
    """
    This only for this project
    """
    # Rearrange the dataset
    ## numerical (17)
    num = ['BT_NM', 'HR_NM', 'RR_NM', 'HB_NM', 'HCT_NM', 'PLATELET_NM', 'WBC_NM', 'PTT1_NM',
           'PTT2_NM', 'PTINR_NM', 'ER_NM', 'BUN_NM', 'CRE_NM', 'BMI', 'age', 'NIHTotal',
           'PPD', ]
    ## category (55)
    cat = ['THD_ID', 'THDA_FL', 'THDH_FL', 'THDI_FL',
           'THDAM_FL', 'THDV_FL', 'THDE_FL', 'THDM_FL', 'THDR_FL', 'THDP_FL',
           'THDOO_FL', 'Gender', 'cortical_ACA_ctr', 'cortical_MCA_ctr', 'subcortical_ACA_ctr',
           'subcortical_MCA_ctr', 'PCA_cortex_ctr', 'thalamus_ctr',
           'brainstem_ctr', 'cerebellum_ctr', 'Watershed_ctr',
           'Hemorrhagic_infarct_ctr', 'cortical_ACA_ctl', 'cortical_MCA_ctl',
           'subcortical_ACA_ctl', 'subcortical_MCA_ctl', 'PCA_cortex_ctl',
           'thalamus_ctl', 'brainstem_ctl', 'cerebellum_ctl', 'Watershed_ctl',
           'Hemorrhagic_infarct_ctl', 'cortical_CT', 'subcortical_CT',
           'circulation_CT',  'watershed_CT', 'Hemorrhagic_infarct_CT',
           'CT_left', 'CT_right', 'CT_find', 'NIHS_1a_in', 'NIHS_1b_in', 'NIHS_1c_in',
           'NIHS_2_in', 'NIHS_3_in', 'NIHS_4_in', 'NIHS_5aL_in', 'NIHS_5bR_in',
           'NIHS_6aL_in', 'NIHS_6bR_in', 'NIHS_7_in', 'NIHS_8_in', 'NIHS_9_in',
           'NIHS_10_in', 'NIHS_11_in', ]
    ## Label (1)
    label = df[['elapsed_class']]
    # imputation
    imp_mode = SimpleImputer(strategy='most_frequent')
    imp_mean = SimpleImputer(strategy='mean')
    df[cat] = imp_mode.fit_transform(df[cat])
    df[num] = imp_mean.fit_transform(df[num])

    df = pd.concat([df[num], df[cat], label], axis=1)

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
    # gradients_sqr = tf.square(gradients)
    # gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=1, keepdims=True)
    # gradients_l2_norm = tf.sqrt(gradients_sqr_sum)
    gradients_l2_norm = tf.norm(gradients, keepdims=True)
    gp = tf.reduce_mean(tf.square((gradients_l2_norm-1.)))

    return gp


def generator_loss(gen_output):
    gen_loss = -tf.reduce_mean(gen_output)
    return gen_loss

def identifiability(gen_data, orig_data):
    
    return 

def rule_constraint(data, sc):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # Reverse data to original values
    dataset = sc.inverse_transform(data.numpy())
    # condition 1: NIHSS total <=42 & >=1
    NIHSS_sub_sum = np.sum(dataset[:, -15:])
    # if sum > 42 or <1 return False (0)
    NIHSS_result = np.where((NIHSS_sub_sum <= 42) & (NIHSS_sub_sum >= 1), 1, 0)
    NIHSS_loss = cross_entropy(tf.ones_like(NIHSS_result), NIHSS_result)

    return NIHSS_loss
