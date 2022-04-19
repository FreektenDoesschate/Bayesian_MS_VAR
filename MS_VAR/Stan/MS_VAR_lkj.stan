data {
int<lower=1> T; 			// Number of observations
int<lower=1> dim; 			// Dimension of observations
matrix[T,dim] y; 			// Observations
int<lower=1> ARdim; 			// Number of AR terms
int<lower=1> nreg; 			// Number of regimes
int<lower = 0, upper = 1> mean_reg; 	// A boolean value indicating if there is a unique mean structure for each regime.
int<lower = 0, upper = 1> sigma_reg; 	// A boolean value indicating if there is a unique covariance structure for each regime.
vector[nreg] Q_alpha; 			// Transition probabilities hyperparameter
real mu_mean; 				// Mean value of the normal prior for the constant means
real mu_sd; 				// Standard deviation of the normal prior for the constant means
real phi_mean; 				// Mean value of the normal prior for the autoregressive coefficients
real phi_sd; 				// Standard deviation of the normal prior for the autoregressive coefficients
real eta; 				// LKJ hyperparameter
real gamma_alpha; 			// Inverse gamma hyperparameter (shape)
real gamma_beta;  			// Inverse gamma hyperparameter (scale)
}

parameters {
simplex[nreg] Q[nreg] ;  					// Transition probabilities
vector[dim] mu[mean_reg ? nreg : 1]; 				// Mean
matrix[dim,dim*ARdim] phi[mean_reg ? nreg : 1]; 		// AR coefficients

vector<lower=0>[dim] vari[sigma_reg ? nreg : 1]; 		// Variance
cholesky_factor_corr[dim] L_corr[sigma_reg ? nreg : 1]; 	//Lower Cholesky factors of the correlation matrices
}

transformed parameters{
vector[dim] sdev[sigma_reg ? nreg : 1];
matrix[dim,dim] L_sigma[sigma_reg ? nreg : 1];			// Covariance matrix for each regime (or single cov function if identical for each regime) 

for(i in 1:(sigma_reg ? nreg : 1)){
	for(j in 1:dim){
		sdev[i,j]=sqrt(vari[i,j]);
	}
	L_sigma[i] = diag_pre_multiply(sdev[i], L_corr[i]); 	// Something to do with cholesky factor (LKJ prior)
}
}

model {
// Priors

for (k in 1:nreg){
	Q[k] ~ dirichlet(Q_alpha); 			// Set prior on transition probability matrix
}

for (k in 1:(mean_reg ? nreg : 1)){
	mu[k] ~ normal(mu_mean,mu_sd); 			// prior on means for each regime	
	to_vector(phi[k]) ~ normal(phi_mean,phi_sd); 	// Phi are the autoregressive coefficients (coefficient that determines how y[t-1], y[t-2] etc relate to y[t])
}

for (k in 1:(sigma_reg ? nreg : 1)){
	vari[k] ~ inv_gamma(gamma_alpha, gamma_beta);	// what does the inverse gamma prior do?
	L_corr[k] ~ lkj_corr_cholesky(eta);		// prior on correlation matrix. 
}
{
    //forward algorithm for computing log p(y|...)
    
	real fwd[nreg,nreg];
    	real alphas[T-ARdim,nreg];
	real py[nreg];
	real logQ[nreg,nreg];		// Log transition matrix
	int n_m = mean_reg ? nreg : 1; 	// number of regimes with distinct mean 
	int n_s = sigma_reg ? nreg : 1; // number of regimes with distinct covariance matrix?
	vector[dim] meanval[n_m];	// mean per regime (or single mean if identical for each regime)
	row_vector[dim*ARdim] ylags; 	// data at timepoint t including lagged data
	
	for(i in 1:nreg){
		for (j in 1:nreg){
			logQ[i,j]= log(Q[i,j]);  // Why is Q, transit prob, now 2 dimensions?
		}
	}
	
	for(i in 1:ARdim){
		ylags[(i-1)*dim+1:i*dim]=y[ARdim+1-i,]; // set data for first timepoint (ie, ARdim+1)
	}
	for(k in 1:n_m){				// this is the actual VAR part, at timepoint 1 in this case
		meanval[k]=mu[k]+phi[k]*ylags'; 	// Meanval is a collection (n=dim) of vectors of length n_m
		// meanval = mean + phi*lagged_observations. Phi are the autoregressive coefficients.
		// meanval are the measured signals for each regime at first timepoint in this case
	}
	
	for(i in 1:nreg){	
		int ii = i; //added by jbb for min fix
		alphas[1,i] = multi_normal_cholesky_lpdf(y[ARdim+1,] | meanval[min(n_m,ii)],L_sigma[min(n_s,ii)]);  // Set alpha for first timepoint? Why conditional on minimal values?
	}
	// alphas are the probabilities at each 'branch' of the forward algorithm tree. 
	// Each alpha contains the probabilities of all nodes below in the tree (due to the markov property)
	
	
    for (t in (ARdim+2):T){						// for each timepoint

		for(i in 1:ARdim){
			ylags[(i-1)*dim+1:i*dim]=y[t-i,];		// set ylags to the observations at t:t-ARdim. ie, data (incl lags) for that timepoint.
		}
		for(k in 1:n_m){					// VAR at timepoint t
			meanval[k]=mu[k]+phi[k]*ylags';			// measured signals (dim dimensions) for each regime is equal to mean of regime + AR coefs*signal 
			
		}
		
		for(i in 1:nreg){				
			int ii = i; //added by jbb for min fix
			py[i] = multi_normal_cholesky_lpdf(y[t,] | meanval[min(n_m,ii)],L_sigma[min(n_s,ii)]); 
			// draw from multivariate correlation prior for data at time t, given the predicted observations based on lags (meanval) and covariance prior? (L_sigma)
			// py = predicted correlation at time t? = emission for each regime?
		}	
		
		for(i in 1:nreg){		// at each timepoint, 
		  for (j in 1:nreg){
			fwd[j,i] = alphas[t-ARdim-1,j] + logQ[j,i] + py[i]; // for each regime-to-regime combination, 
		}
		  alphas[t-ARdim,i]=log_sum_exp(fwd[,i]);
		}
		  
    }
	
	// adding the marginal log-likelihood to Stan target distribution
	
    target += log_sum_exp(alphas[T-ARdim,]);	// Add the alphas for each regime at the final timepoint (final step of forward algorithm)
}
}

generated quantities {
	matrix[dim,dim] sigma[sigma_reg ? nreg : 1];
	int<lower=1,upper=nreg> S[T-ARdim];
	real log_p_S;
  
	for(i in 1:(sigma_reg ? nreg : 1)){
		sigma[i]= multiply_lower_tri_self_transpose(L_sigma[i]); // Recover correlation matrix from cholesky factors
	}
	
   // Viterbi algorithm 
  { 
    int back_ptr[T-ARdim,nreg];
    real best_logp[T-ARdim+1,nreg];
    real best_total_logp;
	real py[nreg];
	real logQ[nreg,nreg];
	int n_m = mean_reg ? nreg : 1;
	int n_s = sigma_reg ? nreg : 1;
	vector[dim] meanval[n_m];
	row_vector[dim*ARdim] ylags;
    
    for(i in 1:nreg){
	best_logp[1,i]=0;
	}
	
	for(i in 1:nreg){
		for (j in 1:nreg){
			logQ[i,j]= log(Q[i,j]);
	}
	}
	
    for (t in (ARdim+1):T) {
	
		for(i in 1:ARdim){
			ylags[(i-1)*dim+1:i*dim]=y[t-i,];
		}
		for(k in 1:n_m){
			meanval[k]=mu[k]+phi[k]*ylags';
		}
			  	  
	    for(i in 1:nreg){
		  int ii = i; //added by jbb for min fix
		  py[i] = multi_normal_cholesky_lpdf(y[t,] | meanval[min(n_m,ii)],L_sigma[min(n_s,ii)]);
		  best_logp[t-ARdim+1,i] = negative_infinity();
	    }	
	  
        for (j in 1:nreg) {
			real logp[nreg];

			for(k in 1:nreg){	
				logp[k] = best_logp[t-ARdim,j] + logQ[j,k] + py[k];
					
				if (logp[k] > best_logp[t-ARdim+1,k]){
					back_ptr[t-ARdim,k] = j;
					best_logp[t-ARdim+1,k] = logp[k];
				}	
			}
        }
    }
	
    log_p_S = max(best_logp[T-ARdim+1]);
    for (k in 1:nreg){
      if (best_logp[T-ARdim+1,k] == log_p_S){
        S[T-ARdim] = k;
	  }
	}
    for (t in 1:(T - ARdim-1)){
      S[T-ARdim-t] = back_ptr[T-ARdim-t+1, S[T-ARdim-t+1]];
	}
  }
}
