#Some Copy past just to understand what's going on !
import numpy as np
from math import ceil,floor
import sys
import matplotlib.pyplot as plt
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%                                                                     %
#%%                   MDestimate3.m                                      %
#%%                                                                     %              
#%%        D. Veitch   P.Abry   P. Chainais                             %
#%%                                                                     %
#%%                           23/09/99                                  %
#%% DV, Melbourne, 1/6/2001                                             %
#%% PA, Lyon, July/10 2002  alpha/zeta bug fix      
#%% PA, Lyon, Oct/21 2002 Add inc to work with increments of order N
#%% PA, lyon, jan04 force gj = 0 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%
#%    This function estimates the scaling exponents zeta_q for a set of moment orders q, using either the 
#%    L2 (energy) preserving normalisation 1/root(a), or the L1 normalisation, 1/a, for the wavelet coefficients.
#%    If more than one q value is asked for, the different exponents are gathered together into the 
#%    `Multiscale Diagram',     ie zeta_q   vs q,  and the
#%    `Linear Multiscale Diagram', zeta_q/q vs q,  where linear behaviour is more easily seen as a straight 
#%    horizontal 	    endline.  The exponents correpond to the scaling of the
#%    partition functions  S_q(j),  via  E[ S_q(j) ] ~ C2^(j zeta_q)
#%
#%    3 Signal Types are possible, for which the analysis differs in the way in which the E[ log(S_q(j) ]
#%    and the Var[log(S_q(j))] are estimated, where each uses weighted regression with an attempted bias correction :
#%    (the calculations are performed directly in wtspec_statlog)
#%     1) Gaussian  Large nj(q): For each q,j: Asymptotic iid based analytic expressions
#%                  Small nj(q): For each q,j: Tabulated values
#%     2) Non-Gaussian. Large nj(q): For each q,j: Semi-parametric estimation for bias and variance (C_j/n_j)
#%             Small nj(q): For each q,j: Same, but fitting using point estimates and extrapolation from larger scales
#%     3) For Alpha Stable and other:  direct estimation for large nj, not implemented 
#%
#%     inc = 1 , computes increments of order N instead of wavelets
#%
#% ***  Usage:    [slope,Vzeta,Q] = MDestimate2(data,N,norm,sigtype,q,j1,j2,printout,inc)
#%
#%--- Routines called Directly :
#% 
#% wtspecq_statlog.m
#%  "[Elogmuqj,Varlogmuqj,nj]=wtspecq_statlog(appro,N,norm,nbvoies,q,distn)"
#% regrespond_det.m
#%  "function [slope,Vzeta,Q]=regrespond_det(q,N,  yj,varj,nj,wtype,  j1,j2,printout);
#%
#%
#%  Input:  data:  the input data (as the obvious practical approximation to the zeroth wavelet "approximation" coeffs)
#%             N:    number of vanishing moments (of the Daubechies wavelet).
#%          norm:    wavelet normalisation convention:  
#%                        1 = L1 [1/scale]       (exponent called  zeta_q),     Zeta_q Diagram,
#%                        2 = L2 [1/sqrt(scale)] (exponent called alpha_q), Logscale_q Diagram
#%       sigtype:  0:  Gaussien, bias et variance of the S_q calculated (in wtspecq_statlot)               (G in title)
#%                 1:  Non-G with finite variance, bias et variance of the S_q estimated (wtspecq_statlot) (NG in title) 
#%                 2:  Alpha-stable  - not implemented  (alpha S in title).
#%                 q:    the powers used for S_q(j)  => can be a vector
#%            j1:    the lower limit of the scales chosen,  1<= j1 <=scalemax-1, can be a vector
#%            j2:    the upper limit of the octaves chosen, 2<= j2 <=scalemax,   can be a vector
#%      printout:  0:   batch mode: no printout, no graphs, only warnings
#%                 1:   graphs and printouts (for each q)
#%                      -- a log-log plot (the LD) is plotted plus the regression line and +- 1.96*std(log[S_q(j)]) 
#%                         around each point, the 95% confidence interval under Gaussian assumptions. 
#%                      -- If q is a vector: The graphs are put into a table,  but no more than "maxtable"
#%                         will be shown (vary this internal parameter if desired)
#%                      -- The MD and the LMD are plotted in a single figure 
#%                      -- the values of H etc are printed.
#%                 2:   single LD and MD only on figure 26, don't plot the LD table, very 
#%                      useful if q very long, too many and too slow to plot
#%                 3:   like printout=1 plus interactive changes of parameters. 
#%                      -- a loop is entered allowing interactive choosing of (sigtype, q, j1,j2) (return to exit)
#%                      -- if q is changed it can only be to a scalar, but the LD table and MD, LMD, will be 
#%                         output for the original q vector if (j1,j2) changed. 
#%
#%  Output:  slope :    estimate of the spectral estimate  zeta (vector)
#%           Vzeta   :    estimate of the variance of slope (vector)
#%           Q       :    confidence indicator;  Q is a vector.
#%
#%----------------------------------------------------------------------------
def MDestimate3(data,N,norm,sigtype,q,j1,j2,printout,inc):

  #%sys.stdout.write('**** In MDestimate \n')

  #%%  Internal parameters
  maxtable = 8    #% don't put more than this many  LD's in the one table, slow and dangerous!
  fsize = 16      #% set font size 

  #%%%% Initialise
  n = len(data)
  nbvoies = floor( log2(n) )
  #%% fix up input q vector
  q = sorted( q[q!=0] )                #% get rid of q=0 if present, keep the rest ordered, make life easy
  lenq = len(q)   
  slop = np.zeros(lenq) 

  firstime = 1
  q_changed = 0
  sigtype_changed = 0
  #% calculate dimensions of table of figures, one per q value
  if printout>0: 
    if lenq < 4:
	rownum = 1
	colnum = lenq
    else:
	colnum = 4 
	rownum = ceil(min(maxtable,lenq)/(1. * colnum))

  #% select the kind of weighting for regression calculation
  if sigtype==0: 
    wtype = 1   #% 2nd order weighting, 1/nj type
  else:
    wtype = 0   #% 0 is uniform weighting,   2 is for data

  #%%%%%%%%%%%%%% beginning of the interactive loop  
  while 1:
    del Q
    
    #%% see if need to run  wtspecq_statlog  this time around
    if firstime | q_changed | sigtype_changed:
      #%  perform the decomposition using Daubechies wavelets, and estimate mu_q(j)
      if sigtype >1:    # %  not yet implemented
	  sys.stdout.write('Sigtype %d Not implemented \n',sigtype)
	  break
      else:
	  if inc == 1:
	      [Elogmuqj, Varlogmuqj, nj,logmuqj] = wtspecq_statlog2(data,N,norm,nbvoies,q,sigtype)  
	  else:
	      [Elogmuqj, Varlogmuqj, nj,logmuqj] = wtspecq_statlog3(data,N,norm,nbvoies,q,sigtype) #  % supports same sig-types

      nbvoies = length(nj)    #%  record ACTual number of octaves available

      if printout:
	sys.stdout.write(' No of points n_j at octave j:   ')
	sys.stdout.write('%d ',nj)
	sys.stdout.write('\n')
	sys.stdout.write(' Number predicted by nj=n*2^j:   ')
	sys.stdout.write('%d ',floor(n*2.^(-(1:nbvoies))) )
	sys.stdout.write('\n\n')
      
	
      firstime = 0
      q_changed = 0
      sigtype_changed = 0

    
    #%%%% Perform the joint parameter estimations and calculate the goodness of fit measure 
    #%--- Check and clean up the inputted j1 and j2, either initially or after interaction 
    if length(j1)<lenq :
      j1 = j1(1)*ones(1,lenq)   #% if not specified, same j1 for all q 
    if length(j2)<lenq :
      j2 =j2(1)*ones(1,lenq) #% if not specified, same j2 for all q 

    j2 = min(j2,nbvoies) #% make sure that j2 cannot exceed the maximum number of octaves actually available 
    j2 = max(j2,2)       #% make sure that j2>1 
    j1 = min(j1,j2-1)    #% make sure that j1<j2 
    
    #%--- For each q value, estimate zeta_q and plot it (and copy plot if lenq>1) 
    for k in np.arange(1,lenq):#% always call this, will create a single q graph if printout>0.  Note doesn't depend on sigtype
      slope[k],Vzeta[:,k],Q[k],aest[k]=regrespond_det2(q(k),N,Elogmuqj(k,:),Varlogmuqj(k,:),nj,wtype, j1(k),j2(k),printout) 
      #%  [slop(k),Vzeta(:,k),Q(k),aest(k)]=regrespond_det2(q(k),N,logmuqj(k,:),Varlogmuqj(k,:),nj,wtype, j1(k),j2(k),printout) ; 
      alphaq(k) = slope(k) #% here alphaq is just stores all the slopes that come out, regardless of norm
      alphq(k) = slop(k)
      if norm == 2:
	  slope(k) = alphaq(k) - q(k)/2 
	  slop(k) = alphq(k) - q(k)/2 
	if printout>0:#set(gca,'FontSize',fsize)
	  if k>1:      #% if lenq>1, lighten up the table, details already in first plot
	    if printout~=2:   #% if want table
	      plt.title('q = '+str(q(k))+',   Q = '+str(Q(k))) ## ?? pq num2str(Q(k),4)
	      plt.xlabel('')  
	  #else: on verra plus tard   #% first plot: fix up title
	    #titlehandle = get(gca,'title');
	    #titlestring = get(titlehandle,'string');   #% get current title string
	    #if  norm == 2  #%  if L2 norm, use alpha_q instead, and call it Logscale Diagram
	      #titlestring = strrep(titlestring,'\zeta','\alpha');
	      #titlestring = strrep(titlestring,'Zeta','Logscale');
	    
	    #if  sigtype==0 sig='G'; elseif sigtype==1 sig='NG'; else sig='\alpha S'; end   
	    #titlestring = [ titlestring, ', ',sig];       #% attach sigtype label
	    #set(titlehandle,'string',titlestring);        #% impose new title
	  #if lenq>1 & printout!=2 & k<=maxtable:  #% if want table, store figure for each, but not too many
	    #tmpname = groupfig(gcf,1,1,99+k)    #% create intermediate plot 
	    #figlist(k) = tmpname                #% keep list of them 

    
    #%--- Multiple q output, group plots into a table and output the MD and LMD 
    ##if printout>0 & lenq>1:#%%% group and plot the stored plots in one figure 
      ##if  printout!=2   #% if want table
	    ##groupfig(figlist,rownum,colnum,300);
	  ##% close([figlist 22])    #% close intermediate graphs and single q output figure

	  
      ###%%% Plot the MD and LMD 
      ##figure(26);  clf   
      ##seuil = 1.9599;

      ###%% MD
      ###%  First add the origin: q=0,  where zeta_0=0, Var[zeta_0]=0
      ##[qplot,index] = sort([0,q]);    #% insert into list in order
      ##lenqplot = length(qplot) ;
      ##zetaplot = [0, slope];          #% insert theoretical zeta of 0
      ##if norm == 2
	      ##zetaplot = [0, alphaq];
      ##end
      ##zetaplot = zetaplot(index);     #% make it correspond to q=0
      ##Vzetaplot = [0, Vzeta];        #% insert theoretical variance of 0
      ##Vzetaplot = Vzetaplot(index);   #% make it correspond to q=0% Calculate confidence intervals
      ###% Calculate confidence intervals
      ##clear CI;
      ##CI(1,:) = zetaplot-seuil*sqrt(Vzetaplot);   #% Gaussian CI's
      ##CI(2,:) = zetaplot+seuil*sqrt(Vzetaplot);
      ###% Determine extremes for plotting
      ###%  mi = min(CI(1,:)) - 0.1;  
      ###%  ma = max(CI(2,:)) + 0.1;
      ##mi = min(min(CI)) - 0.05    #% this works even when q's are negative
      ##ma = max(max(CI)) + 0.05
      ##subplot(1,2,1)
      ##plot(qplot,zetaplot,'o--')
      ##hold on
      ##for k = 1:lenqplot
	  ##plot([qplot(k) qplot(k)], CI(:,k),'b-');
      
      ##axis([min(qplot)-1/2 max(qplot)+1/2 mi ma])
      ##set(gca,'FontSize',fsize)
      ##title(['Multiscale Diagram:  (j_1,j_2) = (',int2str(j1(1)),',  ',int2str(j2(1)),')'])
      ##xlabel('q')
      ##if  norm == 2  % if L2 norm, use alpha_q instead
	##ylabel('\alpha_q','Rotation',0)
      ##else 
	##ylabel('\zeta_q','Rotation',0)
      ##end
      ##grid on; hold off
      
      ###%% LMD
      ###% Calculate confidence intervals
      ##clear CI;
      ##CI(1,:) = (slope-seuil*sqrt(Vzeta))./q   #% Gaussian CI's,   they are scaled by q, just like zeta_q
      ##CI(2,:) = (slope+seuil*sqrt(Vzeta))./q
      ###% Determine extremes for plotting
      ##mi = min(min(CI)) - 0.05    #% this works even when q's are negative
      ##ma = max(max(CI)) + 0.05
      ##subplot(1,2,2)
      ##plot(q,slope./q,'o--')
      ##hold on
      ##for k = 1:lenq
	  ##plot([q(k) q(k)], CI(:,k),'b-');
      ##end
      ##axis([min(q)-1/2 max(q)+1/2 mi ma])
      ##set(gca,'FontSize',fsize)
      ##if  norm == 2  % if L2 norm, use alpha_q instead
	  ##title(['Linear Multiscale Diagram:  h_q=\alpha_q / q - 1/2'])
      ##else 
	  ##title(['Linear Multiscale Diagram:  h_q=\zeta_q / q'])
      ##end
      ##xlabel('q')
      ##ylabel('h_q','Rotation',0)
      ##grid on; hold off
      #%figure(3)
      #%plot(q,slope)
    #% otherwise will just keep the last plot from regrespond_det for a single q (figure 22)

    
    #% Printout a summary of the octaves: in-data/used/available/safe, and the results on H etc...
    #% generate graphs
    if (printout)
      
    sys.stdout.write('******************************************************************\n')
    sys.stdout.write(' Using Sigtype %d :   ',sigtype)
    if sigtype==0
	sys.stdout.write('Gaussian.   [ Using weighted regression with bias correction ]\n\n')
	sys.stdout.write('  Large nj(q): For each q,j: Asymptotic iid based analytic expressions for bias and variances \n')
	sys.stdout.write('  Small nj(q): For each q,j: Tabulated values\n')
    elseif sigtype==1
	sys.stdout.write('Non-Gaussian, finite variance.   [ Using weighted regression with bias correction ]\n\n')
	sys.stdout.write('  Large nj(q): For each q,j: Semi-parametric estimation for bias and variances (C_j/n_j)\n')
	sys.stdout.write('  Small nj(q): For each q,j: Tabulated values as per Gaussian case\n')
    else	
	sys.stdout.write('Alpha Stable and other.   ****  Not implemented **** \n')
	sys.stdout.write('Direct estimation for all q, nj:   Not implemented \n')
    end
    #sys.stdout.write('\n   [ q         j1        j2         Q     exponent ] = \n'),[q',j1',j2',Q',slope']
    sys.stdout.write('******************************************************************\n')

    #%%% AFFICHAGE DES RESULTATS POUR TOUS LES    q     %%%  
    for k=1:lenq
      sys.stdout.write('\n-----------\n')
      sys.stdout.write('| q = %3.1f |\n',q(k))
      sys.stdout.write('-----------\n')	
      sys.stdout.write(' Octaves:     in_data       available     selected\n')
      sys.stdout.write('              1--%3.1f        1--%d         %d--%d \n', log2(n), length(nj),j1(k),j2(k) )
    
      #% Calculate Confidence Intervals 
      #% H  95% two sided gaussian assumption
      sig_level = 5
      seuil = 1.9599
      #%  sig_level quantiles for standard normal (two-sided)
      z1 = sqrt(2) * erfinv(2*   sig_level/2/100  -1) 
      z2 = sqrt(2) * erfinv(2*(1-sig_level/2/100) -1) 
      alpha=alphaq(k) 
      H = slope(k)/q(k);
      CIrad = seuil*sqrt(Vzeta(k))
      HL = H - CIrad/q(k)    #%  currently Vzeta is only a vector Vzeta(q)
      HR = H + CIrad/q(k)
      alH = alpha+CIrad
      alL = alpha-CIrad
      zH  = slope(k)+CIrad
      zL  = slope(k)-CIrad

      #%  Print the output
      sys.stdout.write('MAIN RESULTS ARE:                    alpha           	 H               zeta\n')
      sys.stdout.write('                                     %4.3f        	%4.3f           %7.4f  \n',alpha,H,slope(k) )
      sys.stdout.write('  .95 Conf Int  :              [%4.3f, %4.3f]     [%4.3f, %4.3f]    [%4.3f %4.3f]\n' ,alL,alH,HL,HR,zL,zH)
      sys.stdout.write('        Variance pour zeta  :   %5.4f,     Goodness of fit  Q = %8.6f  \n',Vzeta(k),Q(k))
      #% end of for k=1:lenq
    #%%% FIN AFFICHAGE %%%

    #%%% Take care of interactive parameters, essentially only allow a single new q value at a time
    if printout <3        #% printout = 1 or 2,  not interactive 
      break;              #% nothing more to do, exit printout loop
    else                  #% printout = 3,  Interactive mode, prompt and only print the current figure
      sys.stdout.write('\n  Interactive Mode, if q is changed it can only be a scalar, only one graph will be plotted\n')
      #%% Store previous values
      sigtypeprev=sigtype
      #% Store last q values used (will just be a single value if have already changed q)
      qprev=q
      j1prev=j1
      j2prev=j2
      
      #%%%%  Prompt for new (j1,j2,q) values
      #%%  Sigtype control
      sigtype=input('New sigtype (0 (Gaussian) ou 1 ?      (hit return to exit loop or keep same value) ')
      if isempty(sigtype)  
	  sigtype=sigtypeprev
      else
	  sigtype_changed = 1
      end

      #%%  q control
      changeq=input('Do you want to use other values of the exponents q (hit 1 if yes) ? ')
      if isempty(changeq)
	  q=qprev; 
      else 
	  q=input('New values of the exponents vector q  (hit return to exit loop) ? ')
	  lenq = 1
	  q_changed = 1

      if isempty(q)   #%  Just let me out
	  break
    

      #%%   j1 control
      j1=input('New initial octave j1 ?       (hit return to exit loop or keep same value) ')
      if ( isempty(j1) & ~q_changed )   #% keep the same if choose to AND if q hasn't changed
	  j1=j1prev
      else :if ( isempty(j1) ):    #% want to keep the same but q has changed
	  j1=j1prev(length(qprev))   #% use the existing value for the last q value
 
      #%  if length(j1)<lenq      #% new scalar j1 inputted, but q unchanged, so in fact need a vector
    #%         j1=j1*ones(1,lenq);       % if not specified, same j1 for all q
    #%     end  

      #%%   j2 control
      j2=input('New  final  octave j2 ?       (hit return to exit loop or keep same value) ');
      if ( isempty(j2) & ~q_changed )
	  j2=j2prev;
      elseif ( isempty(j2) )    % want to keep the same but q has changed
	  j2=j2prev(length(qprev)); 
      end  

      if ((sigtype==sigtypeprev) & isempty(changeq) & (j1(1)==j1prev(1)) & (j2(1)==j2prev(1)))
	    break     #% if nothing has changed, exit
      end
    
      clear j1prev j2prev sigtypeprev slope Vzeta
      hold off
      sys.stdout.write('\n**************************************************************************************\n');
      sys.stdout.write('**************************************************************************************\n\n\n');
    end  % 
    #%%% End of interactive mode

    
    else  # % if you can't see the answer there's no point prompting for more values
      break
    end # %end of if printout

  # % end of while

  return slope,Vzeta,Q,Elogmuqj,Varlogmuqj,nj,aest,logmuqj,slop