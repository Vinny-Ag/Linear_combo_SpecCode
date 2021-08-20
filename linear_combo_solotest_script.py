#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 09:51:57 2021

@author: vinny-holiday
"""
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy
import cmath
from scipy import special
from scipy import integrate
from numpy.polynomial.hermite import hermval

'''the number of coefficients used is equal to the number of Hermite polynomials used,
so if you use a single number it will only call H0 the zeroth polynomial'''
# coeff=[0,0,0,0,0,0,0,1]
# coeff2=[1,3]
# coeff1=[1]
# Hv=hermval(1,coeff)
# print('Hermval:',Hv)
# print('')
# print('Coeff length:',len(coeff))

fine_struct=0.0072973525693
au_to_fs=41
hbar_in_eVfs=0.6582119514
ang_to_bohr=1.889725989
# fs_to_Ha=2.418884326505*10.0**(-2.0)
Ha_to_eV=27.211396132
# Ha_to_cm=219474.63
# kb_in_Ha=8.6173303*10.0**(-5.0)/Ha_to_eV
# emass_in_au=5.4857990*10**(-4.0)
# hessian_freqs_to_cm=5140.487125351268
# debye_in_au=0.3934303
# gaussian_to_debye=1.0/6.5005  # Scaling factor translating gaussian output
# 				# of transition dipole moment derivative (Km/mol)^0.5
# 				# to Debye/(ang amu*0.5) 

spectral_window=6
E_adiabatic = 2

num_points = 4000
n= 20
n_max_gs = 20
n_max_ex = 20

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
omega_gs = 0.01599479871665311 #6.61246943e14 Hz
omega_ex = 0.01099445915946515
# omega_gs = 0.010353662727838314 #4.28020304e14 rad (value when not divided by 2pi)

'Hartree units'
adiabatic = 0.0734987 # = 2eV
# bohr_to_1meter_conversion = 1.8897*10**10
# Ha1_to_Joul_conversion = 4.3597*10**-18
# start_point=-0.5702352228173105
# end_point=0.8892357991145113

def spring_const(alpha,D):
# 	alpha_conv=alpha*(bohr_to_1meter_conversion)
# 	D_conv = D*(Ha1_to_Joul_conversion)
	return (alpha**2)*2*D

print('spring constant:',spring_const(alpha_gs,D_gs))

def mode_freq(alpha_gs,D_gs,alpha_ex,D_ex,mu):
	omega_gs = np.sqrt(spring_const(alpha_gs,D_gs)/mu)/2*math.pi
	omega_ex = np.sqrt(spring_const(alpha_ex,D_ex)/mu)/2*math.pi
	return omega_gs,omega_ex

print('omega:',mode_freq(alpha_gs,D_gs,alpha_ex,D_ex,mu))
print('omega:',mode_freq(alpha_gs,D_gs,alpha_ex,D_ex,mu)[0])

def compute_Harm_eval_n(omega,D,n):
	return omega*(n+0.5)-(omega**2/(4.0*D)*(n+0.5)**2.0)

def find_classical_turning_points_morse(n_max_gs,n_max_ex,alpha_gs,alpha_ex,D_gs,D_ex,shift):
	freq_gs = mode_freq(alpha_gs,D_gs,alpha_ex,D_ex,mu)[0]
	freq_ex = mode_freq(alpha_gs,D_gs,alpha_ex,D_ex,mu)[1]
	E_max_gs=compute_Harm_eval_n(freq_gs,D_gs,n_max_gs) # compute the energies for the highest energy morse 
	E_max_ex=compute_Harm_eval_n(freq_ex,D_ex,n_max_ex) # state considered
	# find the two classical turning points for the ground state PES
	point1_gs = math.sqrt(2*E_max_gs/spring_const(alpha_gs,D_gs))
	point2_gs = -math.sqrt(2*E_max_gs/spring_const(alpha_gs,D_gs))
	
	point1_ex = math.sqrt(2*E_max_ex/spring_const(alpha_ex,D_ex))+shift
	point2_ex = -math.sqrt(2*E_max_ex/spring_const(alpha_ex,D_ex))+shift

	# now find the smallest value and the largest value
	start_point=min(point1_gs,point2_gs)
	end_point=max(point1_ex,point2_ex)
	return start_point,end_point

print('start & end point:',find_classical_turning_points_morse(n_max_gs,n_max_ex,alpha_gs,alpha_ex,D_gs,D_ex,shift_ex))
start_point=find_classical_turning_points_morse(n_max_gs,n_max_ex,alpha_gs,alpha_ex,D_gs,D_ex,shift_ex)[0]
end_point=find_classical_turning_points_morse(n_max_gs,n_max_ex,alpha_gs,alpha_ex,D_gs,D_ex,shift_ex)[1]

''' below we establish the classical turning points, but Tim increases their sizes by 10% to account for the tunneling that occurs 
in quantum turning points '''
start_point=start_point+start_point*0.1
end_point= end_point+end_point*0.1
step_x=(end_point-start_point)/num_points
x_range = np.arange(start_point,end_point,step_x)

omega_gs = mode_freq(alpha_gs,D_gs,alpha_ex,D_ex,mu)[0]
omega_ex = mode_freq(alpha_gs,D_gs,alpha_ex,D_ex,mu)[1]

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

def psi_func(x_range,omega,mu,n,shift):
	wavefunc=((num_points,2))
	r_val=x_range*(math.sqrt(mu*omega))
	r_shift_val=shift*(math.sqrt(mu*omega))
	herm_coeffs = np.zeros(n+1)
	herm_coeffs[n] = 1
	norm=1/math.sqrt((2**n)*math.factorial(n))*(mu*omega/math.pi)**(0.25)
	psi = norm*np.exp(-((r_val-r_shift_val)**2)/2)*np.polynomial.hermite.hermval(r_val-r_shift_val,herm_coeffs)
	return psi


# plt.plot(x_range,psi_func(x_range,alpha_gs,D_gs,alpha_ex,D_ex,mu,4,shift_ex))

print('psi',psi_func(x_range,omega_gs,mu,4,shift_ex))
print('psi_shape',np.shape(psi_func(x_range,omega_gs,mu,4,shift_ex)))


def LC_coefficients(x_range,omega_gs,omega_ex,mu,n,shift):
	psi_O = Morse_wavefunc(num_points,start_point,end_point,D_gs,alpha_gs,mu,0,0)
	LC_coeffs = np.zeros(n+1)
	for i in range(n+1):
		LC_coeffs[i] = integrate.simps(psi_func(x_range,omega_gs,mu,i,shift)*psi_O[:,1],x_range,dx=step_x)
	return LC_coeffs

coefficients = LC_coefficients(x_range,omega_gs,omega_ex,mu,n,shift_ex)
print('coefficients:',coefficients)
print('coefficients sum SQ:',sum(coefficients))
print('')


def Linear_combo_wfs(x_range,omega_gs,omega_ex,mu,n,shift):
	coefficients=LC_coefficients(x_range,omega_gs,omega_ex,mu,n,shift)
	LC_func = np.zeros((num_points,n+1))
	for i in range (n+1):
		LC_func[:,i]=psi_func(x_range,omega_gs,mu,i,0)*coefficients[i]
	#print(LC_func)
	return LC_func.sum(axis=1)

print('LC_func_total:',Linear_combo_wfs(x_range,omega_gs,omega_ex,mu,n,shift_ex))
print('')


def transition_energy(omega_ex,D_ex,omega_gs,D_gs,n):
	E_gs=compute_Harm_eval_n(omega_gs,D_gs,n)
	E_ex=compute_Harm_eval_n(omega_ex,D_ex,n)
	return (E_ex-E_gs)+adiabatic

print ('transition energy:',transition_energy(omega_ex,D_ex,omega_gs,D_gs,0))
print('')

def energy_distribution(alpha_gs,D_gs,alpha_ex,D_ex,mu,n):
	max_abs = D_ex-compute_Harm_eval_n(omega_ex,D_ex,0)+transition_energy(omega_ex,D_ex,omega_gs,D_gs,0)#(D_ex+adiabatic)*Ha_to_eV-compute_Harm_eval_n(omega_gs,D_gs,0)*au_to_fs*hbar_in_eVfs#largest possible excitation energy
	min_abs = transition_energy(omega_ex,D_ex,omega_gs,D_gs,0)#0-0 transition
	vib_ex1 = compute_Harm_eval_n(omega_ex,D_ex,1)#+adiabatic*Ha_to_eV
	vib_ex0 = compute_Harm_eval_n(omega_ex,D_ex,0)
	'''doesnt take in to account the decrease in vibrational eigenvalue energy difference as state increases definitely a source of error 
	can we add a decay rate???'''
	E_vib=np.zeros(n)
	for i in range(n):
		E_vib[i]= compute_Harm_eval_n(omega_ex,D_ex,i+1)-compute_Harm_eval_n(omega_ex,D_ex,i) #energy difference between two vibrational harmonic eigenstates
	
	E_trans = np.zeros(n)
	E_range= np.zeros(num_points)
	E_stepx = (max_abs-min_abs)/num_points

	
	#populate E_trans with vibronic transition energies
	for i in range(n):
		if i == 0:
			E_trans[i]=min_abs+0 #the zero position 
		else:
			E_trans[i]=min_abs + i*E_vib[i]
	
	#'populate E_range with energy grid over which to plot spectrum  (in eV)'
	for i in range(num_points):
		E_range[i] = (min_abs-(min_abs*.2))+i*E_stepx #subtracted 20% from min_abs to capture full peak width (min_abs-(min_abs*0.1))
		
	return E_trans,E_range

# E_range = energy_distribution(alpha_gs,D_gs,alpha_ex,D_ex,mu,n)[1]
print ('E_range',energy_distribution(alpha_gs,D_gs,alpha_ex,D_ex,mu,n)[1])
print('E_trans',energy_distribution(alpha_gs,D_gs,alpha_ex,D_ex,mu,n)[0])
print ('space here')
print ('')

def gaussian_setup(alpha_gs,D_gs,alpha_ex,D_ex,mu,n):
	E_range = energy_distribution(alpha_gs,D_gs,alpha_ex,D_ex,mu,n)[1]
	E_trans = energy_distribution(alpha_gs,D_gs,alpha_ex,D_ex,mu,n)[0]
	s = 0.001 #arbitrary gaussian SD, causes linewidth to change
	full_function = np.zeros((num_points,n))
	for i in range(n):
		for k in range(num_points):
			full_function[k,i]=np.exp(-(E_range[k]-E_trans[i])**2/(2*s**2))

	return full_function

print('full_function:',gaussian_setup(alpha_gs,D_gs,alpha_ex,D_ex,mu,n))
# plt.plot(gaussian_setup(alpha_gs,D_gs,alpha_ex,D_ex,mu,n)[:,0])

def plot_setup(x_range,alpha_gs,D_gs,alpha_ex,D_ex,mu,n,shift):
	full_function = gaussian_setup(alpha_gs,D_gs,alpha_ex,D_ex,mu,n)
	overlap_coefs = LC_coefficients(x_range,omega_gs,omega_ex,mu,n,shift)
	overlap_function=np.zeros((num_points,n))
	sumed_function = np.zeros((num_points,1))
	for i in range(n):
		overlap_function[:,i] = full_function[:,i]*overlap_coefs[i]**2
	print('overlap_func',overlap_function)
# 	for i in range(num_points):
# 		sumed_function[i] = sum(overlap_function[i,:])
	sumed_function = overlap_function.sum(axis=1)
	return sumed_function

print('plot_setup',plot_setup(x_range,alpha_gs,D_gs,alpha_ex,D_ex,mu,n,shift_ex)[500])

def absorbance_correction(x_range,alpha_gs,D_gs,alpha_ex,D_ex,mu,n,shift_ex):
	E_range = energy_distribution(alpha_gs,D_gs,alpha_ex,D_ex,mu,n)[1]
	summed_function = plot_setup(x_range,alpha_gs,D_gs,alpha_ex,D_ex,mu,n,shift_ex)
	adjusted_plot=np.zeros((summed_function.shape[0],1))
	for i in range(num_points):
		adjusted_plot[i]= summed_function[i]*2.7347*E_range[i]*Ha_to_eV#*40.0*math.pi**2.0*fine_struct*E_range[i]/(3.0*math.log(10.0))*2.7347
	return adjusted_plot



E_range = energy_distribution(alpha_gs,D_gs,alpha_ex,D_ex,mu,n)[1]
plot_ready_function =absorbance_correction(x_range,alpha_gs,D_gs,alpha_ex,D_ex,mu,n,shift_ex)
print('plot_ready_func:',plot_ready_function.shape)

def write_output(plot_ready_function,E_range):
	spectrum_file = open('./Linear_combo.out', 'w')
	xarray = np.array(E_range*Ha_to_eV)
	yarray = np.array(plot_ready_function)
	data = np.column_stack([xarray,yarray])
	np.savetxt(spectrum_file,data,fmt=['%10.10f','%10.10f'])
	spectrum_file.close()
	return spectrum_file

my_file = write_output(plot_ready_function,E_range)


plt.plot(E_range*Ha_to_eV,plot_ready_function)
plt.xlabel("Energy (eV)")
plt.ylabel("Intensity (arb.)")
plt.show()


def compute_mean_sd_skew(spec):
	# first make sure spectrum has no negative data points:
	counter=0
	while counter<spec.shape[0]:
		if spec[counter,1]<0.0:
			spec[counter,1]=0.0
		counter=counter+1
	step=spec[1,0]-spec[0,0]

	# now compute normlization factor
	norm=0.0
	for x in spec:
                norm=norm+x[1]*step

	mean=0.0
	for x in spec:
		mean=mean+x[0]*x[1]*step
	mean=mean/norm

	sd=0.0
	for x in spec:
		sd=sd+(x[0]-mean)**2.0*x[1]*step
	sd=math.sqrt(sd)/norm

	skew=0.0
	for x in spec:
		skew=skew+(x[0]-mean)**3.0*x[1]*step
	skew=skew/(sd**3.0)
	skew=skew/norm

	return mean,sd,skew
