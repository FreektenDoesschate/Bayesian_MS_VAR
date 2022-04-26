// Stan code for a Markov-switching vector autoregressive (MS-VAR) model.

// MS-VAR model for timeseries {y} of dimension {dim} and timepoints {T}, with {ARdim} lags and {nreg} regimes.

// Regimes can be determined by the (lag)predicted value of {y[t,dims]}, the variance-covariance matrix of {y[t,dims]} or both. 
// These options can be set with {mean_reg} and sigma_reg, respectively. 
// With the {mean_reg} option set to 1, each regime gets its own mean vector {mu} and AR coefficients {phi}.
// With {sigma_reg} set to 1, each regimes gets its own covariance matrix {L_sigma}. 
// Both set to 1, each regime gets a unique mean vector, AR coefs and cov matrix. 

// Using the cholesky factor {eta}, the covariance is modeled as a combination of variances (i.e., amplitude of the signal) and correlation matrix (i.e., cross-correlation). 
// eta > 1, extreme correlations are less likely. eta < 1, extreme correlations are more likely.

// The autoregressive coefficients {phi} all have the same prior Normal(phi_mean, phi_sd).
// {meanval} is the prediction of y[t,] based on a mean vector {mu}, lags {ylags} and AR coefficients {phi}, overall ({mean_reg} == 0) or regemine specific ({mean_reg} == 1)
// {L_sigma} is the covariance matrix which is updated over time, overall ({sigma_reg} == 0) or regemine specific ({sigma_reg} == 1)

// The algorithm is determining emission probabilities {py} for each regime using the lkj-prior, given the predicted {meanval} and {L_sigma} up to that timepoint.

// The transition matrix {Q} is modeled as a vector of dirichlet distributions. 
// The parameters are set using vector {Q_alpha}, which sets each row of Q to a dirichlet with this vector.
// For custom Qs, set dirichlet priors manually in code.

// The Markov-model is then estimated using a forward model, using {alphas}, see https://www.youtube.com/watch?v=9-sPm4CfcD0. 
// {alphas[t,i]} represent the probability that the observed values at time {t} are produced by regime {i}. This object will be updated over time untill the last timepoint. 
// Specifically, first, the intial values of {alphas} and predicted {meanval} are established.
// Then, for each timepoint the {alphas} are determined by the {alphas} at the previous timepoint, the transition probabilities {logQ}, and the emission probabilities {py}.
// This is repeated until the last timepoints. The {alphas} at the last timepoint for each regime are summed. This is the final likelihood (target +=) 

// Something about Viterbi algorithm. Used to backreconstruct which regime is most likely for each timepoint, based on the final alphas?
// {sigma} is the final covariance (or correlation?) matrix (backtransformed from cholesky factor)

// how to add granger causality?



data {
int<lower=1> T; 			// Number of observations
int<lower=1> dim; 			// Dimension of observations
matrix[T,dim] y; 			// Observations
int<lower=1> ARdim; 			// Number of AR terms
int<lower=1> nreg; 			// Number of regimes
int<lower = 0, upper = 1> mean_reg; 	// A boolean value indicating if there is a unique mean structure for each regime. ie, regimes based on prediction of y[t] based on lagged data.
int<lower = 0, upper = 1> sigma_reg; 	// A boolean value indicating if there is a unique covariance structure for each regime. ie, regimes based on covariance between y[t,].
vector[nreg] Q_alpha; 			// Transition probabilities hyperparameter
real mu_mean; 				// Mean value of the normal prior for the constant means
real mu_sd; 				// Standard deviation of the normal prior for the constant means
real phi_mean; 				// Mean value of the normal prior for the autoregressive coefficients
real phi_sd; 				// Standard deviation of the normal prior for the autoregressive coefficients
real eta; 				// LKJ hyperparameter
real gamma_alpha; 			// Inverse gamma hyperparameter (shape), for variance (amplitude). Can be altered to include prior information on amplitudes of seizure phases.
real gamma_beta;  			// Inverse gamma hyperparameter (scale), for variance (amplitude). Can be altered to include prior information on amplitudes of seizure phases.
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

for(i in 1:(sigma_reg ? nreg : 1)){ 				// if sigma_reg > 1, each regime gets its own covariance/correlation matrix
	for(j in 1:dim){
		sdev[i,j]=sqrt(vari[i,j]);
	}
	L_sigma[i] = diag_pre_multiply(sdev[i], L_corr[i]); 	// Covariance matrix for each regime; prepare for cholesky shape
}
}

model {
// Priors

for (k in 1:nreg){
	Q[k] ~ dirichlet(Q_alpha); 			// Set prior on transition probability matrix. Change to impose sequential states.
}

for (k in 1:(mean_reg ? nreg : 1)){
	mu[k] ~ normal(mu_mean,mu_sd); 			// prior on means for each regime	
	to_vector(phi[k]) ~ normal(phi_mean,phi_sd); 	// Phi are the autoregressive coefficients (coefficient that determines how y[t-1], y[t-2] etc relate to y[t])
}

for (k in 1:(sigma_reg ? nreg : 1)){
	vari[k] ~ inv_gamma(gamma_alpha, gamma_beta);	// vari is in L_sigma, determines the diagonal of covariance matrix (variance)
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
		// meanval = predicted value of y at time 1, in this ase
	}
	
	for(i in 1:nreg){	
		int ii = i; //added by jbb for min fix
		alphas[1,i] = multi_normal_cholesky_lpdf(y[ARdim+1,] | meanval[min(n_m,ii)],L_sigma[min(n_s,ii)]);  // get initial emission probabilities
	}
	
	
	
    for (t in (ARdim+2):T){						// for each timepoint

		for(i in 1:ARdim){
			ylags[(i-1)*dim+1:i*dim]=y[t-i,];		// set ylags to the observations at t:t-ARdim. ie, data (incl lags) for that timepoint.
		}
		for(k in 1:n_m){					// VAR at timepoint t
			meanval[k]=mu[k]+phi[k]*ylags';			// measured signals (dim dimensions) for each regime is equal to mean of regime + AR coefs*signal 
		}
		
		for(i in 1:nreg){				
			int ii = i; //added by jbb for min fix
			py[i] = multi_normal_cholesky_lpdf(y[t,] | meanval[min(n_m,ii)],L_sigma[min(n_s,ii)]);  // emission probability at timepoint t
			// for each regime, the probability of observed y, given the predicted observations based on lags (meanval) and covariance between observations (L_sigma)
		}	
		
		for(i in 1:nreg){		// at each timepoint, 
		  for (j in 1:nreg){
			fwd[j,i] = alphas[t-ARdim-1,j] + logQ[j,i] + py[i]; // determine all forward terms at t-1.
			// forward term = alphas at previous timepoint * transit probability (Q) * emission probability (py)
			// everything is in log scale, therefore sum instead of multiply
		}
		  alphas[t-ARdim,i]=log_sum_exp(fwd[,i]); // alpha at time t is equal to sum of all forward terms at t-1. Stored to calc forward term at next timepoint, until final timepoint.
		}
		  
    }
	
	// adding the marginal log-likelihood to Stan target distribution
	
    target += log_sum_exp(alphas[T-ARdim,]);	// Add the alphas for each regime at the final timepoint (final step of forward algorithm), to get the final likelihood
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
