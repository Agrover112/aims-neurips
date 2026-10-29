"""Two-dimensional quadrature adapter and exact-kernel Laplace GP classifier."""
import numpy as np
from scipy.special import expit, logsumexp
from scipy.linalg import cho_factor, cho_solve

BUDGETS=(0,1,3,7,15,31)
WEIGHTS=(.1,.2,.2,.2,.2,.1)

def correction(mc,y,mq,offset_sd=1.,slope_sd=.25):
    nodes,weights=np.polynomial.hermite.hermgauss(21)
    u=np.sqrt(2)*offset_sd*nodes[:,None]
    v=1+np.sqrt(2)*slope_sd*nodes[None,:]
    logits=u[:,:,None]+v[:,:,None]*np.asarray(mc)[None,None,:]
    likelihood=(y*logits-np.logaddexp(0,logits)).sum(-1)
    logw=np.log(weights[:,None]*weights[None,:]/np.pi)+likelihood
    posterior=np.exp(logw-logsumexp(logw))
    return (posterior[:,:,None]*expit(u[:,:,None]+v[:,:,None]*np.asarray(mq)[None,None,:])).sum((0,1))

def kernel(x,z,amplitude,lengthscale,offset_sd):
    distance=np.maximum((x*x).sum(1)[:,None]+(z*z).sum(1)[None,:]-2*x@z.T,0)/x.shape[1]
    return offset_sd**2+amplitude**2*np.exp(-.5*distance/lengthscale**2)

def gp_predict(xc,y,mc,xq,mq,amplitude=1.,lengthscale=1.,offset_sd=1.):
    """Bernoulli logistic likelihood, Laplace posterior, GH predictive integral."""
    diagonal=amplitude**2+offset_sd**2
    if not len(y):
        mean=np.asarray(mq); variance=np.full(len(mq),diagonal)
    else:
        k=kernel(xc,xc,amplitude,lengthscale,offset_sd)+np.eye(len(y))*1e-6
        prior=cho_factor(k,lower=True)
        invk=cho_solve(prior,np.eye(len(y)))
        f=np.asarray(mc).copy()
        def objective(a):
            g=a-mc
            return .5*g@invk@g+np.sum(np.logaddexp(0,a)-y*a)
        for _ in range(40):
            p=expit(f); w=np.maximum(p*(1-p),1e-10)
            gradient=invk@(f-mc)+p-y
            delta=cho_solve(cho_factor(invk+np.diag(w),lower=True),gradient)
            rate=1.; old=objective(f)
            while rate>1e-7 and objective(f-rate*delta)>old:
                rate*=.5
            f-=rate*delta
            if np.max(np.abs(rate*delta))<1e-7:
                break
        p=expit(f); root=np.sqrt(np.maximum(p*(1-p),1e-10))
        b=np.eye(len(y))+root[:,None]*k*root[None,:]
        cross=kernel(xc,xq,amplitude,lengthscale,offset_sd)
        mean=mq+cross.T@cho_solve(prior,f-mc)
        weighted=root[:,None]*cross
        variance=np.maximum(diagonal-np.sum(weighted*cho_solve(cho_factor(b,lower=True),weighted),axis=0),0)
    nodes,weights=np.polynomial.hermite.hermgauss(24)
    return expit(mean[:,None]+np.sqrt(2*variance[:,None])*nodes)@(weights/np.sqrt(np.pi))

def evaluate(predict,tasks,bank):
    rows=[]
    for ep in bank:
        task=tasks[ep['task']]; scores={}
        for budget in BUDGETS:
            c=ep['acquisition'][:budget]; t=ep['targets']
            p=predict(task['items'][c],task['labels'][c],task['items'][t])
            assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
            scores[str(budget)]=float(np.mean((p-task['labels'][t])**2))
        rows.append({'subject':task['subject'],'benchmark':task['benchmark'],'weight':ep['weight'],
                     'by_budget':scores,'brier_alc':sum(w*scores[str(b)] for w,b in zip(WEIGHTS,BUDGETS))})
    weights=[r['weight'] for r in rows]
    macro={b:float(np.mean([r['brier_alc'] for r in rows if r['benchmark']==b])) for b in sorted({r['benchmark'] for r in rows})}
    return {'brier_alc':float(np.average([r['brier_alc'] for r in rows],weights=weights)),
            'by_budget':{str(b):float(np.average([r['by_budget'][str(b)] for r in rows],weights=weights)) for b in BUDGETS},
            'by_benchmark':macro,'benchmark_macro_alc':float(np.mean(list(macro.values()))),'tasks':rows}
