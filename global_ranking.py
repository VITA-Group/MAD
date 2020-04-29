
import numpy as np 
import os

def power_iteration(A, num_simulations):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k


def SRCC():
    import scipy.stats
    model_name_lst = ['vgg16bn', 
		'resnet34', 'resnet101', 
		'wrn-101-2', 'resnext101_32x4d', 'se_resnet101', 
		'senet154',
        'nasnetalarge', 'pnasnet5large',
		'resnext101_32x48d_wsl']

    r_imgnet_acc = ['resnext101_32x48d_wsl','effnetE7','pnasnet5large','nasnetalarge','senet154', 'wrn-101-2','se_resnet101','resnext101_32x4d','resnet101','vgg16bn','resnet34']
    r_50k_mad = ['effnetE7', 'pnasnet5large', 'resnext101_32x48d_wsl', 'senet154', 'nasnetalarge', 'wrn-101-2', 'se_resnet101', 'resnext101_32x4d', 'resnet101', 'vgg16bn', 'resnet34']
    r_168k_mad = ["effnetE7", "resnext101_32x48d_wsl", "senet154", "nasnetalarge", "se_resnet101", "wrn-101-2", "pnasnet5large", "resnext101_32x4d", "vgg16bn", "resnet101", "resnet34"]
    
    for i in range(len(r_imgnet_acc)):
        for j in range(len(model_name_lst)):
            if r_imgnet_acc[i] == model_name_lst[j]:
                r_imgnet_acc[i] = j
                break

    for i in range(len(r_50k_mad)):
        for j in range(len(model_name_lst)):
            if r_50k_mad[i] == model_name_lst[j]:
                r_50k_mad[i] = j
                break

    for i in range(len(r_168k_mad)):
        for j in range(len(model_name_lst)):
            if r_168k_mad[i] == model_name_lst[j]:
                r_168k_mad[i] = j
                break

    print('r_imgnet_acc:', r_imgnet_acc)
    print('r_50k_mad:', r_50k_mad)
    print('r_168k_mad:', r_168k_mad)
    
    print ("srcc:", scipy.stats.spearmanr(r_50k_mad, r_imgnet_acc))
    print ("srcc:", scipy.stats.spearmanr(r_168k_mad, r_imgnet_acc))
    print ("srcc:", scipy.stats.spearmanr(r_50k_mad, r_168k_mad))

if __name__ =='__main__':
    SRCC()