import embanded
import numpy as np
import pytest
import copy
    
def test_ols():
    np.random.seed(1)
    num_obs = 50
    F = [np.random.randn(num_obs,10),np.random.randn(num_obs,10)] 
    X  = np.concatenate(F,axis=1)
    W  = np.concatenate([np.random.randn(10,1), np.zeros((10,1))],axis=0)
    N =  np.random.randn(num_obs,1)*2
    Y = X@W +  N
    clf_em = embanded.EMBanded(num_features=len(F),remove_intercept=True,max_iterations=100,
                                       tau =1e-4,
                                       phi=1e-4,
                                       eta=1e-4,
                                       kappa=1e-4)
    clf_em.fit(F,Y)
    
    D = copy.deepcopy(X)
    D -= D.mean(axis=0)
    D = np.c_[np.ones(num_obs),D]
    
    W_ols = np.linalg.lstsq(D,Y,rcond=None)[0]
    L_em = np.sum((clf_em.W.ravel()-W.ravel())**2)
    L_ols = np.sum((W_ols[1:]-W.ravel())**2)
    
    print('EMB: %0.2f, OLS: %0.2f'%(L_em, L_ols))
    assert L_em<L_ols
    
    
def test_shallow():
    
    np.random.seed(1)
   
    num_obs = 20
    X  = np.random.randn(num_obs,10) + 1000
    X_copy = copy.deepcopy(X)
    

    Y = np.random.randn(num_obs,1)
    Y_copy = copy.deepcopy(Y)

    clf_em = embanded.EMBanded(num_features=1,remove_intercept=True,max_iterations=100,
                                       tau =1e-4,
                                       phi=1e-4,
                                       eta=1e-4,
                                       kappa=1e-4)
    clf_em.fit([X],Y)
    
    assert (X==X_copy).all()
    assert (Y==Y_copy).all()
    
    
    
def test_Omega():
    
    np.random.seed(1)
   
    num_obs = 20
    X  = np.random.randn(num_obs,10) + 1000
    Y = np.random.randn(num_obs,1)
   

    clf_em = embanded.EMBanded(num_features=1,remove_intercept=True,max_iterations=100,
                                       tau =1e-4,
                                       phi=1e-4,
                                       eta=1e-4,
                                       kappa=1e-4)
    clf_em.fit([X],Y)
    
    assert np.isclose(clf_em.Omega,np.eye(10)).all()
    assert np.isclose(clf_em.Omega_inv,np.eye(10)).all()
    
    
    clf_em_smooth = embanded.EMBanded(num_features=1,remove_intercept=True,max_iterations=100,
                                       tau =1e-4,
                                       phi=1e-4,
                                       eta=1e-4,
                                       kappa=1e-4,
                                       h=np.array([5]))
    clf_em_smooth.fit([X],Y)
    
    x_grid = np.arange(10)[None,...]

    Omega_j = (
        1 + np.sqrt(3) * np.abs(x_grid.T - x_grid) / 5
    ) * np.exp(-np.sqrt(3) * np.abs(x_grid.T - x_grid) / 5)
    
    assert np.isclose(clf_em_smooth.Omega,Omega_j).all()
    assert np.isclose(clf_em_smooth.Omega_inv,np.linalg.inv(Omega_j)).all()
   
    
   
def test_offset():
    
    np.random.seed(1)
   
    num_obs = 20
    X  = np.random.randn(num_obs,10) + 100  
    Y = np.random.randn(num_obs,1)

    clf_em = embanded.EMBanded(num_features=1,remove_intercept=True,max_iterations=100,
                                       tau =1e-4,
                                       phi=1e-4,
                                       eta=1e-4,
                                       kappa=1e-4)
    clf_em.fit([X],Y)
    
    assert (clf_em.X_offset==np.mean(X,axis=0)).all()
    assert (clf_em.y_offset==np.mean(Y,axis=0)).all()
    
    
    
    
