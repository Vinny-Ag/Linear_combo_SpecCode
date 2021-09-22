#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 09:51:57 2021

@author: vinny-holiday
"""
import os
import sys
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy
import cmath
from scipy import special
from scipy import integrate
from numpy.polynomial.hermite import hermval
from numba import jit, njit, prange

'''the number of coefficients used is equal to the number of Hermite polynomials used,
so if you use a single number it will only call H0 the zeroth polynomial'''
is_solvent = True
solvent_reorg = 0.0015530494095114032
reorg = solvent_reorg

is_emission = False 
# METHOD = EXACT
integration_points_morse = 4000 #if not set in inputfile, default is 2000 in params.py, same as num_steps
num_points = integration_points_morse

num_steps = 1000 #if not set in inputfile, then default is 1000 in params.py
max_states_morse_gs = 1
n_max_gs = max_states_morse_gs
n_gs = n_max_gs

max_states_morse_ex = 20
n_max_ex = max_states_morse_ex
n_ex = n_max_ex

max_t=300.0
temperature = 100.0
DIPOLE_MOM = [1.0, 0.0, 0.0]
E_adiabatic = 2.0
spectral_window = 6
# CHROMOPHORE_MODEL = MORSE
num_morse_oscillators = 1
# GS_PARAM_MORSE gs_params.dat
# EX_PARAM_MORSE ex_params.dat

D_gs = 0.475217 #2.071824545e-18 J
D_ex = 0.20149 #8.78444853e-19 J

'bohr^-1 units'
alpha_gs = 1.17199002 #6.20190409869e-11 meters^-1
alpha_ex = 1.23719524 #6.54695526318e-11 meters^-1

'mass in a.u.'
mu = 12178.1624678 #reduced mass of CO Tim's value #1.1093555e-26 kg
shift = 0.201444790427
K = shift

grid_n_points = num_points
stdout=open('output_file.out','w')

fine_struct=0.0072973525693
Ha_to_eV=27.211396132
kb_in_Ha=8.6173303*10.0**(-5.0)/Ha_to_eV
fs_to_Ha=2.418884326505*10.0**(-2.0)

solvent_cutoff_freq=0.0001
cutoff_freq = solvent_cutoff_freq


E_adiabatic = E_adiabatic/Ha_to_eV
spectral_window = spectral_window/Ha_to_eV
max_t=max_t/fs_to_Ha

expansion_number = 100


freq_gs=math.sqrt(2.0*D_gs*alpha_gs**2.0/mu)
freq_ex=math.sqrt(2.0*D_ex*alpha_ex**2.0/mu)


@jit(fastmath=True)
def compute_morse_eval_n(omega,D,n):
        return omega*(n+0.5)-(omega*(n+0.5))**2.0/(4.0*D)

def find_classical_turning_points_morse(n_max_gs,n_max_ex,freq_gs,freq_ex,alpha_gs,alpha_ex,D_gs,D_ex,shift):
	E_max_gs=compute_morse_eval_n(freq_gs,D_gs,n_max_gs) # compute the energies for the highest energy morse 
	E_max_ex=compute_morse_eval_n(freq_ex,D_ex,n_max_ex) # state considered

	# find the two classical turning points for the ground state PES
	point1_gs=math.log(math.sqrt(E_max_gs/D_gs)+1.0)/(-alpha_gs)
	point2_gs=math.log(-math.sqrt(E_max_gs/D_gs)+1.0)/(-alpha_gs)

	# same for excited state. Include shift vector
	point1_ex=math.log(math.sqrt(E_max_ex/D_ex)+1.0)/(-alpha_ex)+shift
	point2_ex=math.log(-math.sqrt(E_max_ex/D_ex)+1.0)/(-alpha_ex)+shift

	# now find the smallest value and the largest value
	start_point=min(point1_gs,point2_gs)
	end_point=max(point1_ex,point2_ex)

	return start_point,end_point



def set_absorption_variables(n_max_gs,n_max_ex,freq_gs,freq_ex,alpha_gs,alpha_ex,D_gs,D_ex,K):
    freq_gs=math.sqrt(2.0*D_gs*alpha_gs**2.0/mu)
    freq_ex=math.sqrt(2.0*D_ex*alpha_ex**2.0/mu)

    # calculate number of bound states
    nbound_gs=int((2.0*D_gs-freq_gs)/freq_gs)
    nbound_ex=int((2.0*D_ex-freq_ex)/freq_ex)

    if nbound_gs<n_max_gs:
        n_max_gs=nbound_gs
    if nbound_ex<n_max_ex:
        n_max_ex=nbound_ex

    # now define numerical grid. Find classical turning points
    # on ground and excited state PES

    start_point,end_point=find_classical_turning_points_morse(n_max_gs,n_max_ex,freq_gs,freq_ex,alpha_gs,alpha_ex,D_gs,D_ex,K)
    cl_range=end_point-start_point
    # make sure that the effective qm range is 10% larger than the effective classical range
    # to account for tunneling effects 
    grid_start=start_point-0.05*cl_range
    grid_end=end_point+0.05*cl_range
    grid_step=(grid_end-grid_start)/grid_n_points

    return grid_start,grid_end

start_point, end_point = set_absorption_variables(n_max_gs,n_max_ex,freq_gs,freq_ex,alpha_gs,alpha_ex,D_gs,D_ex,K)

grid_start = start_point
grid_end = end_point

step_x=(grid_end-grid_start)/grid_n_points
x_range = np.arange(grid_start,grid_end,step_x)

# print(x_range)

def compute_wavefunction_n(num_points,start_point,end_point,D,alpha,mu,n,shift):
        # first start by filling array with position points:
        wavefunc=np.zeros((num_points,2))
        lamda=math.sqrt(2.0*D*mu)/(alpha)
        step_x=(end_point-start_point)/num_points
        denom=special.gamma(2.0*lamda-n)
        if np.isinf(denom):
                denom=10e280
        num=(math.factorial(n)*(2.0*lamda-2.0*n-1.0))
        normalization=math.sqrt(num/denom)
        counter=0
        for x in wavefunc:
                x[0]=start_point+counter*step_x
                r_val=(start_point+counter*step_x)*alpha
                r_shift_val=(shift)*alpha
                z_val=2.0*lamda*math.exp(-(r_val-r_shift_val))
                func_val=normalization*z_val**(lamda-n-0.5)*math.exp(-0.5*z_val)*special.assoc_laguerre(z_val,n,2.0*lamda-2.0*n-1.0)
                x[1]=func_val
                counter=counter+1

	# fix normalization regardless of value of denominator to avoid rounding errors
        wavefunc_sq=np.zeros(wavefunc.shape[0])
        wavefunc_sq[:]=wavefunc[:,1]*wavefunc[:,1]
        normalization=integrate.simps(wavefunc_sq,dx=step_x)
        for counter in range(wavefunc.shape[0]):
                wavefunc[counter,1]=wavefunc[counter,1]/math.sqrt(normalization)
        return wavefunc

#LC inserted below

def Harm_wavefunc(x_range,omega,mu,n,shift):
	wavefunc=np.zeros((num_points,2))
	r_val=x_range*(math.sqrt(mu*omega))
	r_shift_val=shift*(math.sqrt(mu*omega))
	herm_coeffs = np.zeros(n+1)
	herm_coeffs[n] = 1
	norm=1/math.sqrt((2**n)*math.factorial(n))*(mu*omega/math.pi)**(0.25)
	#renormalize to prevent error buildup
	counter = 0.0
	for i in range(num_points):
		wavefunc[i,0]=start_point+counter*step_x
		counter = counter+1
	wavefunc[:,1] = norm*np.exp(-((r_val-r_shift_val)**2)/2)*np.polynomial.hermite.hermval(r_val-r_shift_val,herm_coeffs)
	psi_norm = integrate.simps(wavefunc[:,1]**2,dx=step_x)
	wavefunc[:,1] = wavefunc[:,1]/math.sqrt(psi_norm)
	
	return wavefunc



def LC_coefficients(x_range,omega,num_points,start_point,end_point,D,alpha,mu,n,expansion_number,shift):
    morse = compute_wavefunction_n(num_points,start_point,end_point,D,alpha,mu,n,shift)
    LC_coeffs = np.zeros(expansion_number)
    for i in range(expansion_number):
        LC_coeffs[i] = integrate.simps(morse[:,1]*Harm_wavefunc(x_range,omega,mu,i,shift)[:,1],x_range)
    return LC_coeffs


def Linear_combo_wfs(x_range,grid_n_points,grid_start,grid_end,omega,D,alpha,mu,n,expansion_number,shift):
	coefficients=LC_coefficients(x_range,omega,grid_n_points,grid_start,grid_end,D,alpha,mu,n,expansion_number,shift)
	LC_func = np.zeros((num_points,expansion_number))
	LC_func_final = np.zeros((num_points,2))
	for i in range (expansion_number):
		LC_func[:,i]=Harm_wavefunc(x_range,omega,mu,i,shift)[:,1]*coefficients[i]
	LC_func.sum(axis=1)
	counter = 0
	for i in range (num_points):
		LC_func_final[i,0]= start_point+counter*step_x
		counter = counter+1
	LC_func_final[:,1]= LC_func.sum(axis=1)
	return LC_func_final



def LC_func_overlaps(x_range,freq_gs,freq_ex,D_ex,alpha_ex,D_gs,alpha_gs,mu,n_gs,n_ex,expansion_number,shift):
	gs_LC_morse=np.zeros((num_points,n_gs))
	ex_LC_morse=np.zeros((num_points,n_ex))
	gs_ex_Mat=np.zeros((n_gs,n_ex))
	counter=0
	step0=start_point
	step1=start_point+step_x
	for i in range(n_gs):
		gs_LC_morse[:,i]=Linear_combo_wfs(x_range,grid_n_points,grid_start,grid_end,freq_gs,D_gs,alpha_gs,mu,i,expansion_number,0.0)[:,1] #start_point+counter*step_x
	print('gs_LC_Morse',gs_LC_morse)
	
	for j in range(n_ex):
		ex_LC_morse[:,j] = Linear_combo_wfs(x_range,grid_n_points,grid_start,grid_end,freq_ex,D_ex,alpha_ex,mu,j,expansion_number,K)[:,1]
	print('ex_LC_Morse',ex_LC_morse)
	
	for k in range(n_gs):
		for l in range(n_ex):
			gs_ex_Mat[k,l]=integrate.simps(gs_LC_morse[:,k]*ex_LC_morse[:,l],x_range)
	return gs_ex_Mat

wf_overlaps=np.zeros((n_max_gs,n_max_ex))
wf_overlaps_sq=np.zeros((n_max_gs,n_max_ex))
transition_energies=np.zeros((n_max_gs,n_max_ex))
gs_energies=np.zeros(n_max_gs)
boltzmann_fac=np.zeros(n_max_gs)
ex_energies=np.zeros(n_max_ex)
exact_response_func=np.zeros((1,1),dtype=np.complex_)
total_exact_response_func=np.zeros((1,1),dtype=np.complex_)


# calculate transition energy between two specific morse oscillators.
def transition_energy(n_gs,n_ex):
    E_gs=compute_morse_eval_n(freq_gs,D_gs,n_gs)
    E_ex=compute_morse_eval_n(freq_ex,D_ex,n_ex)

    return E_ex-E_gs


@jit(fastmath=True)
def compute_morse_chi_func_t(freq_gs,D_gs,kbT,factors,energies,t):
        chi=0.0+0.0j
        Z=0.0 # partition function
        for n_gs in range(factors.shape[0]):
                Egs=compute_morse_eval_n(freq_gs,D_gs,n_gs)
                boltzmann=math.exp(-Egs/kbT)
                Z=Z+boltzmann
                for n_ex in range(factors.shape[1]):
                        chi=chi+boltzmann*factors[n_gs,n_ex]*cmath.exp(-1j*energies[n_gs,n_ex]*t)

        chi=chi/Z
        return cmath.polar(chi)

def compute_overlaps_and_transition_energies():
    for i in range(n_max_gs):
        gs_energies[i]=compute_morse_eval_n(freq_gs,D_gs,i)
        matrix_overlaps = LC_func_overlaps(x_range,freq_gs,freq_ex,D_ex,alpha_ex,D_gs,alpha_gs,mu,n_gs,n_ex,expansion_number,K)
        for j in range(n_max_ex):
            ex_energies[j]=compute_morse_eval_n(freq_ex,D_ex,j)+E_adiabatic
            transition_energies[i,j]=transition_energy(i,j)
            wf_overlaps[i,j]=matrix_overlaps[i,j]
            wf_overlaps_sq[i,j]=wf_overlaps[i,j]**2.0
    print('EX Energies',ex_energies)
    print('MORSE OVERLAP_Sq MAT',wf_overlaps_sq)
    print('MORSE OVERLAP MAT',wf_overlaps)
    print('Transition energies relative to GS',transition_energies)
    return wf_overlaps_sq,transition_energies

def compute_exact_response_func(factors,energies,freq_gs,D_gs,kbT,max_t,num_steps):
        step_length=max_t/num_steps
        # end fc integral definition
        chi_full=np.zeros((num_steps,3))
        response_func = np.zeros((num_steps, 2), dtype=np.complex_)
        current_t=0.0
        for counter in range(num_steps):
                chi_t=compute_morse_chi_func_t(freq_gs,D_gs,kbT,factors,energies,current_t)
                chi_full[counter,0]=current_t
                chi_full[counter,1]=chi_t[0]
                chi_full[counter,2]=chi_t[1]
                current_t=current_t+step_length
#         print('compute_morse_chi_func_t[1]',chi_full[:,1])
#         print('compute_morse_chi_func_t[2]',chi_full[:,2])
        # now make sure that phase is a continuous function:
        phase_fac=0.0
        for counter in range(num_steps-1):
                chi_full[counter,2]=chi_full[counter,2]+phase_fac
                if abs(chi_full[counter,2]-phase_fac-chi_full[counter+1,2])>0.7*math.pi: #check for discontinuous jump.
                        diff=chi_full[counter+1,2]-(chi_full[counter,2]-phase_fac)
                        frac=diff/math.pi
                        n=int(round(frac))
                        phase_fac=phase_fac-math.pi*n
                chi_full[num_steps-1,2]=chi_full[num_steps-1,2]+phase_fac

	# now construct response function
        for counter in range(num_steps):
                response_func[counter,0]=chi_full[counter,0]
                response_func[counter,1]=chi_full[counter,1]*cmath.exp(1j*chi_full[counter,2])
        print('Compute exact response func')
        print(response_func)
        return response_func

def compute_exact_response(temp,max_t,num_steps):
    kbT=kb_in_Ha*temp
    wf_overlaps_sq = compute_overlaps_and_transition_energies()[0]
    transition_energies = compute_overlaps_and_transition_energies()[1]
    exact_response_func=compute_exact_response_func(wf_overlaps_sq,transition_energies,freq_gs,D_gs,kbT,max_t,num_steps)
    return exact_response_func

def compute_total_exact_response(temp,max_t,num_steps):
    for i in range(num_morse_oscillators):
        exact_response_func = compute_exact_response(temp,max_t,num_steps)
        print('Computed response func!')
        print(exact_response_func)
        if i==0:
            total_exact_response_func = exact_response_func

        else:
            for j in range(total_exact_response_func.shape[0]):
                total_exact_response_func[j,1]=total_exact_response_func[j,1]*exact_response_func[j,1]

    # shift final response function by the adiabatic energy gap
    for j in range(total_exact_response_func.shape[0]):
        total_exact_response_func[j,1]=total_exact_response_func[j,1]*cmath.exp(-1j*E_adiabatic*total_exact_response_func[j,0])
    print('TOTAL_EXACT_RESPONSE_SHAPE',total_exact_response_func.shape)
    return total_exact_response_func

#LC inserted above


# define the maximum number of t points this should be calculated for and the maximum number of steps
def compute_2nd_order_cumulant_from_spectral_dens(spectral_dens,kbT,max_t,steps,stdout):
	q_func=np.zeros((steps,2),dtype=complex)
	stdout.write('\n'+"Computing second order cumulant lineshape function."+'\n')
	stdout.write('\n'+'  Step       Time (fs)          Re[g_2]         Im[g_2]'+'\n')
	step_length=max_t/steps
	step_length_omega=spectral_dens[1,0]-spectral_dens[0,0]
	counter=0
	while counter<steps:
		t_current=counter*step_length
		q_func[counter,0]=t_current
		integrant=integrant_2nd_order_cumulant_lineshape(spectral_dens,t_current,kbT)
		q_func[counter,1]=integrate.simps(integrant[:,1],dx=(integrant[1,0]-integrant[0,0]))   #  give x and y axis
		stdout.write("%5d      %10.4f          %10.4e           %10.4e" % (counter,t_current*fs_to_Ha, np.real(q_func[counter,1]), np.imag(q_func[counter,1]))+'\n')
		counter=counter+1
	return q_func

# fix limit of x-->0, Sign in imaginary term?
@jit(fastmath=True)
def integrant_2nd_order_cumulant_lineshape(spectral_dens,t_val,kbT):
	integrant=np.zeros((spectral_dens.shape[0],spectral_dens.shape[1]),dtype=np.complex_)
	for counter in range(spectral_dens.shape[0]):
		omega=spectral_dens[counter,0]
		integrant[counter,0]=omega
		if counter==0:
			integrant[counter,1]=0.0
		else:
			integrant[counter,1]=1.0/math.pi*spectral_dens[counter,1]/(omega**2.0)*(2.0*cmath.cosh(omega/(2.0*kbT))/cmath.sinh(omega/(2.0*kbT))*(math.sin(omega*t_val/2.0))**2.0+1j*(math.sin(omega*t_val)-omega*t_val))

	return integrant

spectral_dens=np.zeros((1,1))
g2_solvent=np.zeros((1,1))
solvent_response=np.zeros((1,1))


def solvent_spectral_dens(omega_cut,reorg,max_omega,num_steps):
	spectral_dens=np.zeros((num_steps,2))
	step_length=max_omega/num_steps
	counter=0
	omega=0.0
	while counter<num_steps:
		spectral_dens[counter,0]=omega
		# this definition of the spectral density guarantees that integrating the spectral dens yields the 
		# reorganization energy. We thus have a physical motivation for the chosen parameters
		spectral_dens[counter,1]=2.0*reorg*omega/((1.0+(omega/omega_cut)**2.0)*omega_cut)
		omega=omega+step_length
		counter=counter+1
	return spectral_dens



# E_reorg = 0.0
# omega_c = 0.0


def calc_spectral_dens(cutoff_freq,reorg,num_points):
    spectral_dens=solvent_spectral_dens(cutoff_freq,reorg,cutoff_freq*20.0,num_points)
    print('Solvent Spectral Density',spectral_dens)
    return spectral_dens
    
def calc_g2_solvent(spectral_dens,temp,num_points,max_t,stdout):
    stdout.write('Compute the cumulant lineshape function for a solvent bath made up of an infinite set of harmonic oscillators.')
    kbT=kb_in_Ha*temp
    g2_solvent=compute_2nd_order_cumulant_from_spectral_dens(spectral_dens,kbT,max_t,num_points,stdout)
    print('solvent g2 cumulant',g2_solvent)
    return g2_solvent

def calc_solvent_response(g2_solvent,is_emission):
    counter=0
    response_func=np.zeros((g2_solvent.shape[0],2),dtype=complex)
    while counter<g2_solvent.shape[0]:
        response_func[counter,0]=g2_solvent[counter,0].real
        if is_emission:
            response_func[counter,1]=cmath.exp(-np.conj(g2_solvent[counter,1]))
        else:
            response_func[counter,1]=cmath.exp(-g2_solvent[counter,1])
        counter=counter+1
    solvent_response=response_func
    print('SOlVENT RESPONSE',solvent_response.shape)
    return solvent_response

def spectrum_prefactor(Eval,is_emission):
	# prefactor alpha in Ha atomic units
	# Absorption: prefac=10*pi*omega*mu**2*alpha/(3*epsilon_0*ln(10))
	# Emission:   prefac=2*mu**2*omega**4*alpha**3/(3*epsilon_0)
	# note that 4pi*epslion0=1=> epslilon0=1/(4pi)

	prefac=0.0
	if not is_emission:
		# absorption constants
		prefac=40.0*math.pi**2.0*fine_struct*Eval/(3.0*math.log(10.0))
	else:
		# emission constants
		prefac=2.0*fine_struct**3.0*Eval**4.0*4.0*math.pi/3.0

	return prefac

def full_spectrum_integrant(response_func,solvent_response_func,E_val,is_solvent):
    integrant=np.zeros(response_func.shape[0])
    counter=0
#     print('full_spectrum_integ_respfunc',response_func.shape)
#     print('full_spectrum_integ_solventrespfunc',solvent_response_func.shape)
    while counter<integrant.shape[0]:
        if is_solvent:
            integrant[counter]=(response_func[counter,1]*solvent_response_func[counter,1]*cmath.exp(1j*response_func[counter,0]*E_val)).real
        else:
            integrant[counter]=(response_func[counter,1]*cmath.exp(1j*response_func[counter,0]*E_val)).real
        counter=counter+1
    return integrant

def full_spectrum(response_func,solvent_response_func,steps_spectrum,start_val,end_val,is_solvent,is_emission,stdout):
	c1 = np.zeros((steps_spectrum,4))
	spectrum=np.zeros((steps_spectrum,2))
	counter=0
	spectrum_file = open('./Linear_Spectrum_in_Ha.out', 'w')
	Response_func_file = open('./Response_func_and_Spectrum_vince.out', 'w')
	Response_func_file.write('\n'+'Total Chromophore linear response function of the system:'+'\n')
	Response_func_file.write('\n'+'  Step       Time (fs)          Re[Chi]         Im[Chi]'+'\n')
	for i in range(response_func.shape[0]):
		c1[i,0] = i+1
		c1[i,1] = np.real(response_func[i,0])*fs_to_Ha
		c1[i,2] = np.real(response_func[i,1])
		c1[i,3] = np.imag(response_func[i,1])
	np.savetxt(Response_func_file,c1,fmt=['%10.0f','%10.10f','%10.10f','%10.10f'])
	Response_func_file.close()
		#spectrum_file.write("%5d      %10.4f          %10.4e       %10.4e" % (i+1,np.real(response_func[i,0])*fs_to_Ha, np.real(response_func[i,1]), np.imag(response_func[i,1]))+'\n')
		#spectrum_file.write(i+1 np.real(response_func[i,0])*fs_to_Ha np.real(response_func[i,1]) np.imag(response_func[i,1]))
	spectrum_file.write('\n'+'Computing linear spectrum of the system between '+str(start_val*Ha_to_eV)+' and '+str(end_val*Ha_to_eV)+' eV.')
	spectrum_file.write('\n'+'Total linear spectrum of the system:'+'\n')
	spectrum_file.write('Energy (Ha)         Absorbance (Ha)'+'\n')	
	step_length=((end_val-start_val)/steps_spectrum)
	while counter<spectrum.shape[0]:
		E_val=start_val+counter*step_length
		prefac=spectrum_prefactor(E_val,is_emission)
		integrant=full_spectrum_integrant(response_func,solvent_response_func,E_val,is_solvent)
		spectrum[counter,0]=E_val
		spectrum[counter,1]=prefac*(integrate.simps(integrant,dx=response_func[1,0].real-response_func[0,0].real))
		spectrum_file.write("%2.5f          %10.4e" % (spectrum[counter,0], spectrum[counter,1])+'\n') 
		counter=counter+1

	spectrum[:,0]=spectrum[:,0]*Ha_to_eV
	spectrum_file.close()
	return spectrum

def compute_morse_absorption(num_steps,temperature,max_t,is_solvent,is_emission,stdout):
    # first compute solvent response. This is NOT optional for the Morse oscillator, same
    # as in the GBOM
    spectral_dens = calc_spectral_dens(cutoff_freq,reorg,num_points)
    g2_solvent = calc_g2_solvent(spectral_dens,temperature,num_steps,max_t,stdout)
    solvent_response = calc_solvent_response(g2_solvent,is_emission)

    # figure out start and end values over which we compute the spectrum
    # at the moment this is a Hack because we have no expression to analytically 
    # evaluate the average energy gap of the Morse oscillator. 
    E_start=E_adiabatic-spectral_window/2.0
    E_end=E_adiabatic+spectral_window/2.0

    total_exact_response_func=compute_total_exact_response(temperature,max_t,num_steps)
    spectrum=full_spectrum(total_exact_response_func,solvent_response,num_steps,E_start,E_end,is_solvent,is_emission,stdout)
    np.savetxt('Morse_LC_spectrum.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')

linear_spectrum = compute_morse_absorption(num_steps,temperature,max_t,is_solvent,is_emission,stdout)