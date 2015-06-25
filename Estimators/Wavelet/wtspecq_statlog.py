#As this matlab programme is a still a bit obscur for me I decided to copy past
#adapt it in Python formalism and see if darkness get brighter...

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%         wtspecq_statlog2.m
#%
#%         Lyon, 99, Sept. the 24th
#% 
#%         P. Abry and D. Veitch  P. Chainais    
#%
#% DV, Melbourne  2 June 2001
#% PA, Lyon, March 03, Add increments of any order and also output logmuqj =
#% log2(1/n_j \sum_k |d_X(j,k)|^q) 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%   This routine is the equivalent of wtspec.m,  but it calculates the estimates 
#%   at each scale of many moments at once, not just second order.
#%   It also provides options for the calculation of the bias and variance of these
#%   estimates, as the analytical results are not as complete as for the q=2 case.
#%   Note that when q is not 2 we are concerned with multiscaling, and think of
#%   multifractality.  This leads us to choose a different convention for the wavelet
#%   normalisation, 1/a rather than 1/root(a), which changes the qth order exponent by -q/2.
#%   Here is can choose either convention according to `norm'.
#%
#% ***  Usage:        [Elogmuqj,Varlogmuqj,nj,logmuqj]=wtspecq_statlog2(appro,N,norm,nbvoies,q,distn,inc) ;
#%
#%--- Routines called Directly :
#%      rlistcoefdaub.m   "h = rlistcoefdaud(N)"
#%      logstat.m         "[LogE,Var,Elog,ndecoupe] = logstat(X,printout)"
#%--- Stored data
#%      varndq_gauss_tab  contains estimates of the E and Var of the mean of N variables~|N(0,1)|^q 
#%  Nlim            1x11  (used once)     critical value of nj(q), below which need to use tabulation
#%  Ntab            1x12  (used once)     values of N tabulated 
#%  nbreal          1x1    (not used)     number of realisations used in the estimation
#%  qtab            1x11  (used 3 times)  values of q tabulated:  [0.5 1 1.5 2 2.5 3 3.5 4 5 6]
#%  biaisgauss_tab 11x12  (used once)     calculated bias,      matrix  (#qtab) x (#Ntab)
#%  vargauss_tab   11x12  (used once)     calculated variance
#%
#%
#%   Input:  appro:    data to be analysed
#%		N:    number of Vanishing Moments of Daubechies Wavelets
#%            norm:    wavelet normalisation convention:  1 = L1,  2 = L2 
#%	  nbvoies:    number of octaves
#%		q:    vectors of q values 
#%           distn:    Hypothesis on the data distribution or `singal type':
#%                     0: Gauss,  1: Non-G finite var, 2: AlphaStable or other
#%
#%  Output: Elogmuqj   : estimation for  log E(1/nj\sum_k |d_X(j,k)|^q) (not E[log( )], as have corrected bias)	
#%	   Varlogmuqj : estimation for the variance of the r.v.  log (1/nj\sum_k |d_X(j,k)|^q)
#%	   nj         : # of available (unpolluted by border effects)  wavelet coef at each octave
#%
#%-----------------------------------------------------------------------
 def wtspecq_statlog2(appro,N,norm,nbvoies,q,distn):

  #%--- Constant
  loge = log2(exp(1))

  #%--- Initialize
  q = sort(q);                          #% keep the q ordered, make life easy
  lenq = length(q);
  n = length(appro) ; 
  %--- Coef of filters
  g0 = [ 1 -1 ] ;
  g1 = g0 ;
  for k = 2:1:N
      g1 = conv(g1,g0) ;
  end
  g1 = g1./sqrt( sum(g1.*g1) ) ;
  lh = 1 ;
  lg = length(g1) ;

  %--- Predict the max # of octaves available given N, and take the min with what asked for
  %nbvoies = min( fix(log2(n/(2*N+1))), nbvoies)         %  never end up with 0
  %nbvoies = min( fix(log2(n/(2*N+1))) + 1 , nbvoies);    %    can end up with 0 (for N>2)
  nbvoies = min( fix(log2(n/(2*N+1)))  , nbvoies);    %   safer, casadestime having problems

  %%--- Distn  dependent initialisation
  if distn==0   %% GAUSSIEN
    %--- Load tabulated values of Expectation and variances for |N(0,1)|^q. for Gaussian case 
    load varndq_gauss_tab    
    % q control:  those asked for not neccesarily tabulated, generate substitutes qctab
    for  k=1:lenq  
	[minvalue minindex] = min( abs(q(k)-qtab) );    % find index of closest q value
	qctab(k) = qtab(minindex);                      % use that value
    end
    if sum(q==qctab) ~=lenq
	fprintf('*** Warning, some q values aren''t tabulated, substituting:  original q were: \n')
	q
	fprintf('Substitutes for tabulated part are:\n')
	qctab
    end
  elseif  distn==1  %% Not-Gaussian, but finite variance,  using asymptotic forms  ~C/nj 
    %--- Choose estimation method for the constant ( C_4(j) + 2 )
    parametric = 0 ;   % Linear Regression for the smart estimate of C_4 at each j using subblocks
    if  ~parametric    % or just do a direct point estimate of C_4(j) 
      fprintf('*** Reminder, doing direct estimation of C_4 rather than regression  ***\n')
    end
  end


  %%%--- Compute the WT, calculate statistics
  njtemp = n;    %  this is the the appro, changes at each scale
  for j=1:nbvoies	
	  %-- Phase 1, get the coefficients at this scale
	  convolue = conv(appro,g1) ;
	  decime = convolue(lg:2:end-lg+1) ;   % details at scale j. decime always becomes empty near the end, 
	  nj(j) = length(decime);            % number of coefficients
	  if nj(j) == 0                      % forget this j and exit if no coefficients left!
	    fprintf('In wtspecq_statlog, oops!  no details left at scale %d! \n',j)
	    break
	  end
	  
	  %-- Phase 2, calculate and store relevant statistics, do this for each q 
	  for  k=1:lenq    %  Loop on the values of q
	    qc = q(k) ;
	    absdqk = abs(decime).^qc ;       % the absolute powers are the new base quantities 
	    logmuqj = log2(mean(absdqk));    % Calculate  log[ S_q(j) ]    ( in L1 language ) 
	    
	    %%% Now calculate the output quantities:  Elogmuqj  and  Varlogmuqj
	    if distn == 0  %% GAUSSIEN
	      k0 = find(qctab(k) == qtab) ;                           % find index of qctab in qtab (will exist)
	      if  nj(j) < Nlim(k0)     %  if nj too small, need to use tabulation 
		fprintf('! Using tabulated values for |Gaussian|^q distribution:  q,j = %4.1f , %d \n',qc,j)
		[minvalue minindex] = min(abs(log2(nj(j))-log2(Ntab))); % choose nearest tabulated nj
		Varlogmuqj(k,j) =   vargauss_tab(k0,minindex)./nj(j);   % grab precalculated values at nearest (q,j)
		gj              = biaisgauss_tab(k0,minindex)./nj(j);   % what about scaling this to real var??
	      else  % Large nj(q): For each q,j: Asymptotic iid based analytic expressions
		% V = 2( 1+C_4/2 ) = 2 + C_4 = [ E[|d|^2q] / (E[|d|^q)^2 ]  - 1,  can calculate if d Gaussian
		V  = sqrt(pi)*gamma(qc+1/2)/gamma((qc+1)/2)^2 - 1;     
		Varlogmuqj(k,j) =  loge^2 * V /nj(j) ;
		gj              = -loge/2 * V /nj(j) ;
	      end
	      Elogmuqj(k,j) = logmuqj - gj ; 
	    elseif  distn==1  %% Not-Gaussian, but finite variance
	      if  parametric   % use a parametric model for bias and variance, and fit from data
		  [LogEj,Varj] = logstat(absdqk,0);   % try to estimate by splitting into blocks
		  if  Varj == 0  % this can happen, is set by logstat as a sign of not enough data
		    if j>1 
		      Varj = 2 * Varlogmuqj(k,j-1);   % if no variance info at j, guess it is twice that at j-1
		    else  #% Very small signal, j-1 doesn't exist, have to estimate directly
		      Varj = loge^2 * (std(absdqk)^2/mean(absdqk)^2 ) /nj(j);  % estimate C_4 + 2 directly
		    end
		  end
		  Varlogmuqj(k,j) = Varj ;  % this is true, an estimate of the Var of the log
		  Elogmuqj(k,j)   = LogEj;  % Beware confusing notation, if bias corrected, this is log(E[muqj])...
	      else   %  Asymptotic formula with direct estimate of C_4 from data
		  V = std(absdqk)^2/mean(absdqk)^2 ;      % estimate  C_4 + 2  directly
		  Varlogmuqj(k,j) =  loge^2 * V /nj(j) ;  % as before, asymptotic formulae
		  gj              = -loge/2 * V /nj(j) ;
		      % Modif PA --- le 11/01/2004 pour forcer gj = 0 
		      % Elogmuqj(k,j) = logmuqj - gj ;   
		  Elogmuqj(k,j) = logmuqj ;   
	      end
	    else  %% Infinite variance, Alpha stable or other
		fprintf('! Distribution type 2 not implemented ! \n'); 
	    end
	    %%% Adjust for wavelet normalisation ,  affects the E term only 
	    if norm == 1  % L1 normalisation, = 1/a
		Elogmuqj(k,j) = Elogmuqj(k,j) - j*qc/2 ; 
	    end          % else, L2 normalisation, = 1/root(a)  
	  end   
	
	  %-- prepare for Phase 1 at next coarser scale
	  clear convolue decime absdqk 
	  % convolue = conv(appro,hh1) ;
	  % appro = convolue(nl:2:njtemp);% new approximation, filtre meme taille que pour detail=> memes effets de bord
      appro = sqrt(2) * appro(lh:2:length(appro)) ; 
	  njtemp = length(appro) ;
	  clear convolue 
  return Elogmuqj,Varlogmuqj,nj,logmuqj