
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

def temp():
    model_name_lst = [
        'vgg16bn','resnet34', 'resnet101', 'wrn-101-2', 'resnext101_32x4d', 'se_resnet101', 
        'senet154','nasnetalarge', 'pnasnet5large', 'resnext101_32x48d_wsl', 'effnetE7'
    ]

    # acc_lst = [73.37, 73.3, 77.37,78.84,78.188,78.396,81.304,82.7,82.9,85.4,84.4]
    acc_lst = [73.36, 73.31, 77.37, 78.85, 78.19, 78.4, 81.3, 82.51, 82.74, 85.44, 84.48]


    N = len(acc_lst)
    judgement_mat = np.full((N,N), np.NaN)
    for i in range(N):
        for j in range(i+1,N):
            judgement_mat[i,j] = acc_lst[i] / acc_lst[j]
        for j in range(0,i):
            judgement_mat[i,j] = 1/judgement_mat[j,i]
        judgement_mat[i,i] = 1

    # print(judgement_mat)

    primary_eigen_vector = power_iteration(judgement_mat, 100)

    print(primary_eigen_vector)
    print(np.dot(judgement_mat, primary_eigen_vector) / primary_eigen_vector)

    score_lst = []
    for i, (element, acc) in enumerate(zip(primary_eigen_vector, acc_lst)):
        subjective_score = element.real
        score_lst.append(subjective_score)
        print('model %d: PEV element %.4f, acc %.2f' % (i, subjective_score, acc))

    sorted_idx = np.argsort(score_lst).tolist()
    sorted_idx.reverse()

    order_str = ''
    order_name_str = ''
    for i in sorted_idx:
        if i == sorted_idx[-1]:
            order_str += '%d' % i
            order_name_str += '%s' % model_name_lst[i]
        else:
            order_str += '%d > ' % i
            order_name_str += '%s > ' % model_name_lst[i]
    print('order by judgement matrix:')
    print(order_str)
    print(order_name_str)


    sorted_idx = np.argsort(acc_lst).tolist()
    sorted_idx.reverse()

    order_str = ''
    order_name_str = ''
    for i in sorted_idx:
        if i == sorted_idx[-1]:
            order_str += '%d' % i
            order_name_str += '%s' % model_name_lst[i]
        else:
            order_str += '%d > ' % i
            order_name_str += '%s > ' % model_name_lst[i]
    print('order by pure acc:')
    print(order_str)
    print(order_name_str)

def temp_rank_imgnet():
    '''
    MAD top-30 ranking on imagenet 50k val set.
    '''
    model_name_lst = [
        'vgg16bn','resnet34', 'resnet101', 'wrn-101-2', 'resnext101_32x4d', 'se_resnet101', 
        'senet154','nasnetalarge', 'pnasnet5large', 'resnext101_32x48d_wsl', 'effnetE7'
    ]

    A = np.load('temp_imagenet_A.npy')
    B = np.full(A.shape, np.nan)

    n, _ = A.shape

    for i in range(n):
        B[i,i] = 1
        for j in range(i+1,n):
            B[i,j] = A[i,j] / A[j,i]
            B[j,i] = 1/B[i,j]

    primary_eigen_vector = power_iteration(B, 100)

    idx_sorted = np.argsort(primary_eigen_vector).tolist() # ascending
    print('idx_sorted:', idx_sorted)
    idx_sorted = idx_sorted[::-1] # descending
    print('idx_sorted:', idx_sorted)
    sorted_lst = []
    for i in idx_sorted:
        sorted_lst.append(model_name_lst[i])
    
    print(sorted_lst) # descending

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
    # temp_rank_imgnet()
    SRCC()