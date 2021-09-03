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
# coeff=[0,0,0,0,0,0,0,1]
# coeff2=[1,3]
# coeff1=[1]
# Hv=hermval(1,coeff)
# print('Hermval:',Hv)
# print('')
# print('Coeff length:',len(coeff))


solvent_cutoff_freq = 0.0001
solvent_reorg = 0.00015530494095114032
is_emission = False
is_solvent = True
num_morse_oscillators = 1
max_t = 300
temp = 50
fine_struct=0.0072973525693
au_to_fs=41
hbar_in_eVfs=0.6582119514
ang_to_bohr=1.889725989
Ha_to_eV=27.211396132
fs_to_Ha=2.418884326505*10.0**(-2.0)

# Ha_to_cm=219474.63
kb_in_Ha=8.6173303*10.0**(-5.0)/Ha_to_eV
# emass_in_au=5.4857990*10**(-4.0)
# hessian_freqs_to_cm=5140.487125351268
# debye_in_au=0.3934303
# gaussian_to_debye=1.0/6.5005  # Scaling factor translating gaussian output
# 				# of transition dipole moment derivative (Km/mol)^0.5
# 				# to Debye/(ang amu*0.5) 

kbT=kb_in_Ha*temp
spectral_window=6
adiabatic = 2

num_points = 4000
expansion_number=125 #the number of HO's used for the linear combination method
# n=30 #the number of Morse states to build with HO's
gs_morse = 1 #number of gs Morse wavefunctions that will overlap with excited states, should be dependent on boltzmann distribution
ex_morse = 20

n_max_gs = gs_morse
n_max_ex = ex_morse

'Hartree units (a.u.)' # a.u. = hartree
D_gs = 0.475217 #2.071824545e-18 J
D_ex = 0.20149 #8.78444853e-19 J

'bohr^-1 units'
alpha_gs = 1.17199002 #6.20190409869e-11 meters^-1
alpha_ex = 1.23719524 #6.54695526318e-11 meters^-1

'mass in a.u.'
# 1 au = 9.10938356e-31 kg for electron
mu = 12178.1624678 #reduced mass of CO Tim's value #1.1093555e-26 kg
#mu = 12590.76119 #reduced mass of CO

'bohr units'
shift_ex = 0.201444790427
shift_ex_angstroms = 0.10659999232835039
shift_gs = 0.0

'frequency in a.u.'
# omega_gs = 0.01599479871665311 #6.61246943e14 Hz
# omega_ex = 0.01099445915946515
# omega_gs = 0.010353662727838314 #4.28020304e14 rad (value when not divided by 2pi)


'Hartree units'
E_adiabatic = adiabatic/Ha_to_eV
E_spectral_window = spectral_window/Ha_to_eV
# bohr_to_1meter_conversion = 1.8897*10**10
# Ha1_to_Joul_conversion = 4.3597*10**-18
# start_point=-0.5702352228173105
# end_point=0.8892357991145113





def spring_const(alpha,D):
# 	alpha_conv=alpha*(bohr_to_1meter_conversion)
# 	D_conv = D*(Ha1_to_Joul_conversion)
	return (alpha**2)*2*D

# print('spring constant:',spring_const(alpha_gs,D_gs))

# def mode_freq(alpha_gs,D_gs,alpha_ex,D_ex,mu):
# 	omega_gs = np.sqrt(spring_const(alpha_gs,D_gs)/mu)/2*math.pi
# 	omega_ex = np.sqrt(spring_const(alpha_ex,D_ex)/mu)/2*math.pi
# 	return omega_gs,omega_ex

def set_absorption_variables(D_gs,alpha_gs,D_ex,alpha_ex,mu):
	omega_gs=math.sqrt(2.0*D_gs*alpha_gs**2.0/mu)
	omega_ex=math.sqrt(2.0*D_ex*alpha_ex**2.0/mu)
	return omega_gs,omega_ex


# print('omega:',mode_freq(alpha_gs,D_gs,alpha_ex,D_ex,mu))
# print('omega:',mode_freq(alpha_gs,D_gs,alpha_ex,D_ex,mu)[0])

# def compute_Harm_eval_n(omega,D,n):
# 	return omega*(n+0.5)-(omega**2/(4.0*D)*(n+0.5)**2.0)

omega_gs,omega_ex = set_absorption_variables(D_gs,alpha_gs,D_ex,alpha_ex,mu)


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

###############################################################################

# print('start & end point:',find_classical_turning_points_morse(n_max_gs,n_max_ex,omega_gs,omega_ex,alpha_gs,alpha_ex,D_gs,D_ex,shift_ex))
start_point,end_point=find_classical_turning_points_morse(n_max_gs,n_max_ex,omega_gs,omega_ex,alpha_gs,alpha_ex,D_gs,D_ex,shift_ex)

''' below we establish the classical turning points, but Tim increases their sizes by 10% to account for the tunneling that occurs 
in quantum turning points '''
start_point=start_point+start_point*0.1
end_point= end_point+end_point*0.5
step_x=(end_point-start_point)/num_points
x_range = np.arange(start_point,end_point,step_x)

###############################################################################


def Morse_wavefunc(num_points,start_point,end_point,D,alpha,mu,n,shift):
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


# plt.plot(x_range,psi_func(x_range,alpha_gs,D_gs,alpha_ex,D_ex,mu,4,shift_ex))

# print('psi',Harm_wavefunc(x_range,omega_gs,mu,n,0))
# print('psi_shape',Harm_wavefunc(x_range,omega_gs,mu,n,0).shape)

# print('Morse wf',Morse_wavefunc(num_points,start_point,end_point,D_gs,alpha_gs,mu,n,0))
# print('Morse_wf_shape',Morse_wavefunc(num_points,start_point,end_point,D_gs,alpha_gs,mu,n,0).shape)

def LC_coefficients(x_range,omega,num_points,start_point,end_point,D,alpha,mu,n,expansion_number,shift):
    morse = Morse_wavefunc(num_points,start_point,end_point,D,alpha,mu,n,shift)
    LC_coeffs = np.zeros(expansion_number)
    for i in range(expansion_number):
        LC_coeffs[i] = integrate.simps(morse[:,1]*Harm_wavefunc(x_range,omega,mu,i,shift)[:,1],x_range)
    return LC_coeffs

Coefficients = LC_coefficients(x_range,omega_ex,num_points,start_point,end_point,D_ex,alpha_ex,mu,30,expansion_number,shift_ex)
# print('coefficients:',Coefficients)
# print('coefficients sum SQ:',sum(Coefficients**2))
# print('')

def Linear_combo_wfs(x_range,omega,D,alpha,mu,n,expansion_number,shift):
	coefficients=LC_coefficients(x_range,omega,num_points,start_point,end_point,D,alpha,mu,n,expansion_number,shift)
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

# print('LC_func_total:',Linear_combo_wfs(x_range,omega_gs,D_gs,alpha_gs,mu,0,expansion_number,0.0))
# print('')
# Morse_LC_wf_gs = Linear_combo_wfs(x_range,omega_gs,D_gs,alpha_gs,mu,0,expansion_number,0.0)
# Morse_LC_wf_ex = Linear_combo_wfs(x_range,omega_ex,D_ex,alpha_ex,mu,2,expansion_number,shift_ex)
# plt.plot(Morse_LC_wf_gs)
# plt.plot(Morse_LC_wf_ex)

#new function


def LC_func_overlaps(x_range,omega_gs,omega_ex,D_ex,alpha_ex,D_gs,alpha_gs,mu,gs_morse,ex_morse,expansion_number,shift):
	gs_LC_morse=np.zeros((num_points,gs_morse))
	ex_LC_morse=np.zeros((num_points,ex_morse))
	gs_ex_Mat=np.zeros((gs_morse,ex_morse))
	counter=0
	step0=start_point
	step1=start_point+step_x
	for i in range(gs_morse):
		gs_LC_morse[:,i]=Linear_combo_wfs(x_range,omega_gs,D_gs,alpha_gs,mu,i,expansion_number,0.0)[:,1] #start_point+counter*step_x
	print('gs_LC_Morse',gs_LC_morse)
	
	for j in range(ex_morse):
		ex_LC_morse[:,j] = Linear_combo_wfs(x_range,omega_ex,D_ex,alpha_ex,mu,j,expansion_number,shift)[:,1]
	print('ex_LC_Morse',ex_LC_morse)
	
	for k in range(gs_morse):
		for l in range(ex_morse):
			gs_ex_Mat[k,l]=integrate.simps(gs_LC_morse[:,k]*ex_LC_morse[:,l],x_range)
	return gs_ex_Mat




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
	while counter<integrant.shape[0]:
		if is_solvent:
			integrant[counter]=(response_func[counter,1]*solvent_response_func[counter,1]*cmath.exp(1j*response_func[counter,0]*E_val)).real
		else:
			integrant[counter]=(response_func[counter,1]*cmath.exp(1j*response_func[counter,0]*E_val)).real
		counter=counter+1
	return integrant



def full_spectrum(response_func,solvent_response_func,steps_spectrum,start_val,end_val,is_solvent,is_emission):
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
	np.savetxt(Response_func_file,c1,fmt=['%10.10f','%10.10f','%10.10f','%10.10f'])
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

# define the maximum number of t points this should be calculated for and the maximum number of steps
def compute_2nd_order_cumulant_from_spectral_dens(spectral_dens,kbT,max_t,steps):
    outfile = open('2nd_ord_cum_from_spec_den.dat','w')
    q_func=np.zeros((steps,2),dtype=complex)
    outfile.write('\n'+"Computing second order cumulant lineshape function."+'\n')
    outfile.write('\n'+'  Step       Time (fs)          Re[g_2]         Im[g_2]'+'\n')
    step_length=max_t/steps
    step_length_omega=spectral_dens[1,0]-spectral_dens[0,0]
    counter=0
    while counter<steps:
        t_current=counter*step_length
        q_func[counter,0]=t_current
        integrant=integrant_2nd_order_cumulant_lineshape(spectral_dens,t_current,kbT)
        q_func[counter,1]=integrate.simps(integrant[:,1],dx=(integrant[1,0]-integrant[0,0]))   #  give x and y axis
        outfile.write("%5d      %10.4f          %10.4e           %10.4e" % (counter,t_current*fs_to_Ha, np.real(q_func[counter,1]), np.imag(q_func[counter,1]))+'\n')
        counter=counter+1
    outfile.close()
    return q_func


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


spectral_dens=np.zeros((1,1))
g2_solvent=np.zeros((1,1))
solvent_response=np.zeros((1,1))

def calc_spectral_dens(num_points):
    spectral_dens=solvent_spectral_dens(solvent_cutoff_freq,solvent_reorg,solvent_cutoff_freq*20.0,num_points)
    print('Solvent Spectral Density',spectral_dens)
    return spectral_dens


def calc_g2_solvent(temp,num_points,max_t):
    spectral_dens = calc_spectral_dens(num_points)
    kbT=kb_in_Ha*temp
    g2_solvent=compute_2nd_order_cumulant_from_spectral_dens(spectral_dens,kbT,max_t,num_points)
    print('solvent g2 cumulant', g2_solvent)
    return g2_solvent
    
def calc_solvent_response(is_emission):
    g2_solvent = calc_g2_solvent(temp,num_points,max_t)
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
    print('SOLVENT RESPONSE', solvent_response)
    return solvent_response

# this function brings together the response function in its most recognizable form
# in terms of theoretical representations, returns resp func in polar coordinates
@jit(fastmath=True)
def compute_morse_chi_func_t(omega_gs,D_gs,kbT,factors,energies,t):
        chi=0.0+0.0j
        Z=0.0 # partition function
        for n_gs in range(factors.shape[0]):
                Egs=compute_morse_eval_n(omega_gs,D_gs,n_gs)
                boltzmann=math.exp(-Egs/kbT)
                Z=Z+boltzmann
                for n_ex in range(factors.shape[1]):
                        chi=chi+boltzmann*factors[n_gs,n_ex]*cmath.exp(-1j*energies[n_gs,n_ex]*t)

        chi=chi/Z	
        print('compute_morse_chi_func_t',cmath.polar(chi))
        return cmath.polar(chi)


def compute_exact_response_func(factors,energies,omega_gs,D_gs,kbT,max_t,num_points):
        step_length=max_t/num_points
        # end fc integral definition
        chi_full=np.zeros((num_points,3))
        response_func = np.zeros((num_points, 2), dtype=np.complex_)
        current_t=0.0
        for counter in range(num_points):
                chi_t=compute_morse_chi_func_t(omega_gs,D_gs,kbT,factors,energies,current_t)
                chi_full[counter,0]=current_t
                chi_full[counter,1]=chi_t[0]
                chi_full[counter,2]=chi_t[1]
                current_t=current_t+step_length
        #print('compute_morse_chi_func_t',chi_full)
        # now make sure that phase is a continuous function:
        phase_fac=0.0
        for counter in range(num_points-1):
                chi_full[counter,2]=chi_full[counter,2]+phase_fac
                if abs(chi_full[counter,2]-phase_fac-chi_full[counter+1,2])>0.7*math.pi: #check for discontinuous jump.
                        diff=chi_full[counter+1,2]-(chi_full[counter,2]-phase_fac)
                        frac=diff/math.pi
                        n=int(round(frac))
                        phase_fac=phase_fac-math.pi*n
                chi_full[num_points-1,2]=chi_full[num_points-1,2]+phase_fac

	# now construct response function
        for counter in range(num_points):
                response_func[counter,0]=chi_full[counter,0]
                response_func[counter,1]=chi_full[counter,1]*cmath.exp(1j*chi_full[counter,2])
        print('Compute exact response func',response_func)
        return response_func


# calculate transition energy between two specific morse oscillators.
def transition_energy(n_gs,n_ex):
	E_gs=compute_morse_eval_n(omega_gs,D_gs,n_gs)
	E_ex=compute_morse_eval_n(omega_ex,D_ex,n_ex)
	return E_ex-E_gs


wf_overlaps=np.zeros((gs_morse,ex_morse))
wf_overlaps_sq=np.zeros((gs_morse,ex_morse))
boltzmann_fac=np.zeros((gs_morse))
transition_energies=np.zeros((gs_morse,ex_morse))
gs_energies=np.zeros(int(gs_morse))
ex_energies=np.zeros(int(ex_morse))

exact_response_func=np.zeros((1,1),dtype=np.complex_)
total_exact_response_func=np.zeros((1,1),dtype=np.complex_)

def compute_exact_response(temp,omega_gs,omega_ex,max_t,num_steps):
	kbT=kb_in_Ha*temp
	for i in range(gs_morse):
		gs_energies[i]=compute_morse_eval_n(omega_gs,D_gs,i)
		for j in range(ex_morse):
			ex_energies[j]=compute_morse_eval_n(omega_ex,D_ex,j)+E_adiabatic
			transition_energies[i,j]=transition_energy(i,j)
	print('EX_Energies',ex_energies)
	print('Transition Energy',transition_energies)
	wf_overlaps=LC_func_overlaps(x_range,omega_gs,omega_ex,D_ex,alpha_ex,D_gs,alpha_gs,mu,gs_morse,ex_morse,expansion_number,shift_ex)
	wf_overlaps_sq=wf_overlaps**2.0
	exact_response_func=compute_exact_response_func(wf_overlaps_sq,transition_energies,omega_gs,D_gs,kbT,max_t,num_steps)
	print('LC MORSE OVERLAPS',wf_overlaps_sq)
	return exact_response_func


# def compute_total_exact_response(temp,max_t,num_points):
# 	total_exact_response_func = compute_exact_response(temp,omega_gs,omega_ex,max_t,num_points)
# 	print('Computed response func!')
# 	print(total_exact_response_func)
# 	for j in range(total_exact_response_func.shape[0]):
# 		total_exact_response_func[j,1]=total_exact_response_func[j,1]*cmath.exp(-1j*E_adiabatic*total_exact_response_func[j,0])
# 	
# 	# shift final response function by the adiabatic energy gap
# 	for j in range(total_exact_response_func.shape[0]):
# 		total_exact_response_func[j,1]=total_exact_response_func[j,1]*cmath.exp(-1j*E_adiabatic*total_exact_response_func[j,0])
# 	return total_exact_response_func

def compute_total_exact_response(temp,max_t,num_steps):
	for i in range(num_morse_oscillators):
		exact_response_func=compute_exact_response(temp,omega_gs,omega_ex,max_t,num_steps)
		print('Computed response func!')
		print(exact_response_func)
		if i==0:
			total_exact_response_func=exact_response_func
			print('TERF',total_exact_response_func)
		else:
			for j in range(total_exact_response_func.shape[0]):
				total_exact_response_func[j,1]=total_exact_response_func[j,1]*exact_response_func[j,1]
	# shift final response function by the adiabatic energy gap
	for j in range(total_exact_response_func.shape[0]):
		total_exact_response_func[j,1]=total_exact_response_func[j,1]*cmath.exp(-1j*E_adiabatic*total_exact_response_func[j,0])
	return total_exact_response_func

def compute_morse_absorption(is_emission,temp,num_points,max_t):
	# first compute solvent response. This is NOT optional for the Morse oscillator, same
	solvent_response = calc_solvent_response(is_emission)
	# figure out start and end values over which we compute the spectrum
	# at the moment this is a Hack because we have no expression to analytically 
	# evaluate the average energy gap of the Morse oscillator. 
	E_start=E_adiabatic-E_spectral_window/2.0
	E_end=E_adiabatic+E_spectral_window/2.0

	# exact solution to the morse oscillator
	total_exact_response_func = compute_total_exact_response(temp,max_t,num_points)
	print('total exact response func')
	print(total_exact_response_func.shape)
	print(total_exact_response_func)
	spectrum=full_spectrum(total_exact_response_func,solvent_response,num_points,E_start,E_end,True,False)
	np.savetxt('Morse_exact_spectrum.dat', spectrum, header='Energy (eV)      Intensity (arb. units)')


linear_absorption = compute_morse_absorption(False,temp,num_points,max_t)


