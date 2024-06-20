/* -----------------------------------------------------------------
 2D Cahn-Hilliard equation - Spinodal decomposition
 Developed by University of Ljubljana, Faculty of Mechanical Engineering
 Laboratory for Internal Combustion Engines and Electromobility - LICeM

 Developers: Igor Mele, Klemen Zelič, Tomaž Katrašnik;
 * -----------------------------------------------------------------*/

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>


#include <ida/ida.h>                          /* prototypes for IDA fcts., consts.    */
#include <nvector/nvector_serial.h>           /* access to serial N_Vector            */
#include <sunmatrix/sunmatrix_dense.h>        /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h>        /* access to dense SUNLinearSolver      */
#include <sunnonlinsol/sunnonlinsol_newton.h> /* access to Newton SUNNonlinearSolver  */
#include <sundials/sundials_types.h>          /* defs. of realtype, sunindextype      */
#include <sundials/sundials_math.h>           /* defs. of SUNRabs, SUNRexp, etc.      */
#include <sunlinsol/sunlinsol_spgmr.h>        /* access to spgmr SUNLinearSolver      */
#include <sunlinsol/sunlinsol_spfgmr.h>       /* access to spfgmr SUNLinearSolver     */
#include <sunlinsol/sunlinsol_spbcgs.h>       /* access to SPBCGS SUNLinearSolver     */
#include <sunlinsol/sunlinsol_sptfqmr.h>      /* access to SPTFQMR SUNLinearSolver    */
#include <sunlinsol/sunlinsol_pcg.h>          /* access to PCG SUNLinearSolver        */

#if defined(SUNDIALS_EXTENDED_PRECISION)
#define GSYM "Lg"
#define ESYM "Le"
#define FSYM "Lf"
#else
#define GSYM "g"
#define ESYM "e"
#define FSYM "f"
#endif

#define VALUE_max(x, y) (((x) > (y)) ? (x) : (y)) //return higher number of the x and y
#define VALUE_min(x, y) (((x) < (y)) ? (x) : (y)) //return lower number of the x and y

/* Solver selection */
#define USE_solver_type 1   //[=0] direct dense linear solver SUNLINSOL_DENSE using a user-supplied Jacobian
                            //[=1] Krylov linear solver SUNLINSOL_SPGMR using only the diagonal elements of the Jacobian
                            //[=2] Krylov linear solver SUNLINSOL_SPFGMR using only the diagonal elements of the Jacobian
                            //[=3] Krylov linear solver SUNLINSOL_SPBCGS using only the diagonal elements of the Jacobian
                            //[=4] Krylov linear solver SUNLINSOL_SPTFQMR using only the diagonal elements of the Jacobian
                            //[=5] Krylov linear solver SUNLINSOL_PCG using only the diagonal elements of the Jacobian


/* Problem Constants */
#define ZERO RCONST(0.0)
#define ONE  RCONST(1.0)

#define R           8.3144621	    //gas constant [J/(mol*K)]
#define T           300.0           //temperature [K]
#define NA          6.0221367e23    //Avogadro number [-]
#define F_CONST     96485.3329      //Faraday constant [As/mol]
#define PI_CONST    3.14159265359   //pi constant

#define N_CV_x      50              //number of control volumes in x-direction [-]
#define N_CV_y      50              //number of control volumes in y-direction [-]
#define NOUT_       1001            //number of iterations [-]
#define dt_         0.001           //time step [s]
#define dt_MAX      1000.0          //maximal allowed time step when using variable time step [s]

#define L_x         100e-9          //domain size in x-direction [m]
#define L_y         100e-9          //domain size in y-direction [m]
#define D_xx_       8.0e-16         //diffusion constant in x-direction [m^2/s]
#define D_yy_       8.0e-16         //diffusion constant in y-direction [m^2/s]
#define Omega_      (4.48*R*T/NA)   //regular solution parameter [J]
#define Kappa_xx_   (4.72e-10)      //strain constant in x-direction [J/m]
#define Kappa_yy_   (4.72e-10)      //strain constant in y-direction [J/m]
#define cm          22800.0         //maximal value for concentration [mol/m^3]
#define c0          (0.5*cm)        //initial value for dimensional concentration [mol/m^3]
#define rndCAmp_    (0.01*cm)       //random amplitude of perturbation around initial dimensional concentration [mol/m^3]
#define dp_         25e-9           //particle thickness in z-direction [m]
#define alpha_a_    0.5             //anodic transfer coefficient [-]
#define alpha_c_    0.5             //cathodic transfer coefficient [-]
#define i0_         0.16            //exchange current density in BV pre-factor [A/m^2]
#define frequency_  1.0             //frequency of the base waveform upon which pulses of Li source are generated [Hz]
#define threshold_  0.5             //threshold on the sinusoidal curve to switch between different modes of pulses (+, -, 0), value between [-1, +1]
#define pulseTime_  0.1             //duration of the pulse [s]
#define relaxTime_  0.1             //duration of relaxation in-between pulses
#define C_DL_       0.2             //double layer capacitance [F/m^2]
#define B_strain_   (0.2375e9)      //strain [Pa]
#define C_rate_     0.0             //C-rate by which source of Li into the 2D domain is defined [1/h]


//relative and absolute tolerances, definition W i = 1/[rtol * |y_i| + atol_i]
#define RTOL        (1.0e-10)  //relative tolerance [-]
#define ATOL_C      (1.0e-10)  //absolute tolerance for non-dimensional concentration [-]
#define ATOL_mu     (1.0e-8)  //absolute tolerance for non-dimensional chemical potential [-]
#define ATOL_dPhi   (1.0e-8)  //absolute tolerance for non-dimensional potential difference [-]

//set relative and absolute tolerances for the first few (N_dt_update_tol) timesteps to help simulation to start from random noise
#define RTOL_INIT       (1.0e-11)
#define ATOL_C_INIT     (1.0e-11)
#define ATOL_mu_INIT    (1.0e-1)
#define ATOL_dPhi_INIT  (1.0e-1)

//values to trigger stop of simulation
#define stationary_level_mu_MAX 1e-15  //maximal value for stationary level for chemical potential

//various options to switch on/off [0/1]
#define USE_variable_timestep       0   //use variable length of time step based on the number of iterations during one time step
#define USE_restart_simulation      0   //use option to restart the simulation from the saved restart file
#define USE_write_animation_files   1   //use option to enable writing of the files intended for animation purposes
#define USE_update_tols_after_n_dts 0   //use option to update tolerances from *_INIT to desired values after few timesteps to help the simulation to start
#define USE_constant_initial_C_mu   0   //use constant initial random distribution of concentration and chemical potential, i.e. without randomized seed
#define USE_IVC_calculation         1   //use option to calculate corrected initial value conditions

//various options to switch between
#define USE_run_mode            0   //[=0] single simulation run of the 2D domain
                                    

#define USE_source_term         0   //[=0] do not use source term;
                                    //[=2] S0 * BV_exp;
                                    //[=3] S0*sqrt(x(1-x)) * BV_exp;
                                    //[=4] S0*sqrt(x(1-x)exp(omega*(1-2x))) * BV_exp;

#define USE_linearised_source   1   //[=0] use normal exponential equation for the source term
                                    //[=1] use linearised source term equation
                                    //[=2] use quadratic/second order source term equation
                                    //[=-1] use piecewise function (exponential equation for the source term + linear extrapolation)

#define USE_pulse_shape         0   //[=0] use constant source term
                                    //[=1] pulses of Li source (always in the same direction, either + or -, depends on the overpotential sign) with relaxation periods in-between
                                    //[=2] pulses of Li source in both directions with relaxation periods in-between
                                    //[=3] read C-rate profile form external file
                                    //[=4] construct pulses based on the pulseTime, relaxTime and C_rate
                                    //[=5] sinusoidal current excitation with chosen frequency

#if USE_source_term > 0 //define use of double layer model only if source term is activated and dPhi is being calculated
    #define USE_double_layer_model  1   //use double layer model
#else
    #define USE_double_layer_model  0   //do not use double layer model
#endif // USE_source_term


//frequency for writing results
#define writeStep_state_vector          2000000 //number of time steps between writing time dependent state vector results
#define writeStep_species_conservation  2000 //number of time steps between writing time dependent species conservation results
#define writeStep_average_mu            2000 //number of time steps between writing time dependent species conservation results
#define writeStep_stationary_level_mu   2000 //number of time steps between writing time dependent values of stationary level of chemical potential
#define writeStep_restart_file          2000 //number of time steps between writing restart file
#define writeStep_Crate_profile         2000 //number of time steps between writing C-rate profile
#define writeStep_potential_difference  2000 //number of time steps between writing potential difference results
#define writeStep_animation_files       2000 //number of time steps between writing animation files
#define writeStep_clock_timestep        1000  //number of time steps between writing duration of the individual timestep


//frequency for calling certain functions
#define step_calculate_stationary_level_mu 1 //number of time steps between calculation of stationary level for chemical potential

//constants not intended to be changed
#define N_var_tot       2   //total number of state variables in each control volume in the problem [C mu]
#define N_var_LSC_tot   1   //total number of state variables related to the calculation of the lithium source control based on the selection of source term [dPhi]
#define N_var_derivs    (N_var_tot + N_var_LSC_tot) //total number of variables to calculate partial derivatives over
#if USE_source_term > 0 //define number of active state variables related to the calculation of the lithium source control based on the selection of source term
    #define N_var_LSC_act   1   // [dPhi]
#else
    #define N_var_LSC_act   0   // [-]
#endif // USE_source_term

#define N_CV_pos        (5+N_var_LSC_tot)   //number of control volume positions [2D = P E W N S] and additional dependent variables [dPhi] (=5+1)
#define N_LSC_pos       1   //number of control volume positions for the lithium source control equation [= P] (=1)
#define NEQ   (N_CV_x*N_CV_y*N_var_tot + N_var_LSC_act) //dimension of the problem
#define N_argc          5   //number of command line arguments +1 for ./main
#define Len_filename    256 //number of chars to be allocated for filenames
#define N_MAX_restarts  100 //maximal number of restarts before exiting the simulation
#define N_dt_update_tol 10  //number of initial time-steps to keep TOL_INIT and then change to default values to help the simulation start
#define N_var_results   3   //number of variables to be stored in the results structure array [simTime Iapp dPhi]

//Solver constants
#define USE_DENSE 0     //[=0] direct dense linear solver SUNLINSOL_DENSE using a user-supplied Jacobian
#define USE_SPGMR 1     //[=1] Krylov linear solver SUNLINSOL_SPGMR using only the diagonal elements of the Jacobian
#define USE_SPFGMR 2    //[=2] Krylov linear solver SUNLINSOL_SPFGMR using only the diagonal elements of the Jacobian
#define USE_SPBCGS 3    //[=3] Krylov linear solver SUNLINSOL_SPBCGS using only the diagonal elements of the Jacobian
#define USE_SPTFQMR 4   //[=4] Krylov linear solver SUNLINSOL_SPTFQMR using only the diagonal elements of the Jacobian
#define USE_PCG 5       //[=5] Krylov linear solver SUNLINSOL_PCG using only the diagonal elements of the Jacobian

/* Macro to define dense matrix elements, indexed from 0. */
#define IJth(A,i,j) SM_ELEMENT_D(A,i,j)


/* Structures */
typedef struct{
    //dimensional properties
    double Lx;          //length of the domain in the x-direction [m]
    double dx;          //width of the control volume in the x-direction [m]
    double Ly;          //length of the domain in the y-direction [m]
    double dy;          //width of the control volume in the y-direction [m]
    double dt;          //time step [s]
    double Omega;       //regular solution parameter [J]
    double C0;          //initial concentration [mol/m^3]
    double Cm;          //maximal concentration [mol/m^3]
    double D_xx;        //diffusion constant in the x-direction [m^2/s]
    double D_yy;        //diffusion constant in the x-direction [m^2/s]
    double Kappa_xx;    //strain in the x-direction [J/m]
    double Kappa_yy;    //strain in the y-direction [J/m]
    double rndCAmp;     //random amplitude around initial dimensional concentration [mol/m^3]
    double dp;          //particle thickness in z-direction [m]
    double i0;          //exchange current density in BV pre-factor [A/m^2]
    double S0;          //constant pre-factor in the source term [mol/m^3 s]
    double alpha_c;     //cathodic transfer coefficient [-]
    double alpha_a;     //anodic transfer coefficient [-]
    double S_TOT;       //total source of Li into the 2D domain [mol/m^3 s]
    double C_rate;      //C-rate representation of the total source of Li into the 2D domain [converted from 1/h to 1/s]
    double tmp_C_rate;  //temporary variable for storing C-rate representation of the total source of Li into the 2D domain [1/h]
    double pulseTime;   //duration of the pulse [s]
    double relaxTime;   //duration of the relaxation phase between the pulses [s]
    double eventTime;   //time of the next event [s]
    double C_DL;        //double layer capacitance [F/m^2]
    double B_strain;    //strain [Pa]
    double tau;         //time constant in the 2D domain [s]

    //non-dimensional properties
    double ndn_Kappa_xx;    //non-dimensional strain in the x-direction [-]
    double ndn_Kappa_yy;    //non-dimensional strain in the y-direction [-]
    double ndn_Chi_xx;      //non-dimensional constant in x-direction [-]
    double ndn_Chi_yy;      //non-dimensional constant in y-direction [-]
    double ndn_Omega;       //non-dimensional regular solution parameter [-]
    double ndn_dt;          //non-dimensional time step [-]
    double ndn_dx;          //non-dimensional width of the control volume in the x-direction [-]
    double ndn_dy;          //non-dimensional width of the control volume in the y-direction [-]
    double ndn_C0;          //non-dimensional initial concentration [-]
    double ndn_rndCAmp;     //non-dimensional random amplitude around initial non-dimensional concentration [-]
    double ndn_S0;          //non-dimensional constant pre-factor in the source term [-]
    double ndn_S_TOT;       //non-dimensional total source of Li into the 2D domain [-]
    double ndn_C_rate;      //non-dimensional C-rate representation of the total source of Li into the 2D domain [-]
    double ndn_pulseTime;   //non-dimensional duration of the pulse [-]
    double ndn_relaxTime;   //non-dimensional duration of the relaxation phase between the pulses [-]
    double ndn_eventTime;   //non-dimensional time of the next event [-]
    double ndn_C_DL;        //non-dimensional double layer capacitance [-]
    double ndn_B_strain;    //non-dimensional strain [-]

} properties;


typedef struct{
    char *filename_species_conservation;
    char *filename_jacobi;
    char *filename_matrix_concentration_start;
    char *filename_matrix_concentration_end;
    char *filename_matrix_chem_potential_start;
    char *filename_matrix_chem_potential_end;
    char *filename_parameters_and_statistic;
    char *filename_stationary_level_mu;
    char *filename_restart_file;
    char *filename_input_Crate_profile;
    char *filename_output_Crate_profile;
    char *filename_potential_difference;
    char *filename_clock_timestep;
    char *filename_animation_concentration;
    char *filename_animation_chem_potential;
    char *filename_animation_supporting_data;
    char *filename_animation_time_data;
    char *filename_average_chemical_potential;

} pathsFilenames;

typedef struct{
    double *sv_CV;      //state vector [number of state variables]
    double ***jac_CV;   //local jacobi (partial derivative evaluations) terms [number of state variables][number of state variables][position P E W N S]
    double *atval_CV;   //local absolute tolerance vector [number of state variables]
    double *idval_CV;   //local vector specifies algebraic/differential components in the y vector [number of state variables]
    int pos_CV_P;       //integer position of the CENTRAL control volume [-]
    int pos_CV_E;       //integer position of the EAST control volume related to the current central control volume [-]
    int pos_CV_W;       //integer position of the WEST control volume related to the current central control volume [-]
    int pos_CV_N;       //integer position of the NORTH control volume related to the current central control volume [-]
    int pos_CV_S;       //integer position of the SOUTH control volume related to the current central control volume [-]
    double S_ndn;       //non-dimensional source term of Li in each control volume [-]
    double *dS_ndn_dSVar;//partial derivatives of non-dimensional source term over state variables [number of state variables]
    double xCV;         //x-coordinate of the control volume [m]
    double ndn_xCV;     //non-dimensional x-coordinate of the control volume [-]
    double yCV;         //y-coordinate of the control volume [m]
    double ndn_yCV;     //non-dimensional y-coordinate of the control volume [-]
} control_volume;


typedef struct{
    double *sv_LSC;      //state vector [number of state variables]
    double ***jac_LSC;   //local jacobi (partial derivative evaluations) terms [number of state variables][number of state variables][position P E W N S]
    double *atval_LSC;   //local absolute tolerance vector [number of state variables]
    double sum_S_ndn_CV; //sum of all non-dimensional source terms of Li across all control volumes
    double *sum_dS_ndn_dSVar; //sum of all partial derivatives of non-dimensional source term over state variables [number of state variables]

} lithium_source_control;


typedef struct{
    double S_DL_ndn;        //non-dimensional source in the domain due to the double layer [-]
    double *dS_DL_ndn_dSVar;//partial derivatives of non-dimensional source due to the double layer over state variables [number of state variables]

} double_layer_model;


typedef struct{
    double *time_data;      //array containing time data
    double *Crate_data;     //array containing C-rate data
    int dataIndx;           //data index
    int dataLength;         //data length, i.e. number of lines in the input file
    int delaySwitch;        //delay switch to adopt change of the C-rate after one time step after event was triggered

} input_data;


typedef struct{
    control_volume *CV; //control volume structure
    properties PRP; //properties structure
    pathsFilenames PFN; //path filenames structure
    lithium_source_control LSC; //lithium source control structure
    input_data INDAT;   //input data structure
    double_layer_model DLM; //double layer model structure
    int pos_SV_C;       //integer position of the concentration in the state vector [-]
    int pos_SV_mu;      //integer position of the chemical potential in the state vector [-]
    int pos_LSC_dPhi;   //integer position of the potential difference in the state vector [-]
    int pos_jac_P;      //integer position of the partial derivative in local jacobian matrix of rhs function over variable in the CENTRAL control volume [-]
    int pos_jac_E;      //integer position of the partial derivative in local jacobian matrix of rhs function over variable in the EAST control volume [-]
    int pos_jac_W;      //integer position of the partial derivative in local jacobian matrix of rhs function over variable in the WEST control volume [-]
    int pos_jac_N;      //integer position of the partial derivative in local jacobian matrix of rhs function over variable in the NORTH control volume [-]
    int pos_jac_S;      //integer position of the partial derivative in local jacobian matrix of rhs function over variable in the SOUTH control volume [-]
    int pos_jac_dPhi;   //integer position of the partial derivative in local jacobian matrix of rhs function over dPhi variable [-]
    int pos_deriv_C;    //integer position of the partial derivative over concentration in the array of partial derivatives [-]
    int pos_deriv_mu;   //integer position of the partial derivative over chemical potential in the array of partial derivatives [-]
    int pos_deriv_dPhi;     //integer position of the partial derivative over potential difference in the array of partial derivatives [-]
    double *master_RHS;     //complete right hand side vector
    double ndn_simTime;     //non-dimensional simulation time at a given moment[-]
    double clock_solver;    //elapsed time for solver [s]
    double clock_timestep;  //elapsed time for a single time step [s]
    char *resultPath;       //relative path to the selected folder for storing simulation results
    char *animationResultPath; //relative path to the selected folder for storing simulation results for animation purposes
    double stationary_level_mu; //measure for stationary level of chemical potential
    int i_USE_source_term;  //index of the selected ansatz for the source term
    double ndn_avg_C;       //average non-dimensional concentration in the 2D domain [-]
    double ndn_avg_mu;      //average non-dimensional chemical potential in the 2D domain [-]
    int pulse_relax_ID;     //ID number for selecting between pulse and relaxation period [-]
    double *JacDiag;        //array of values of diagonal elements of the Jacobi matrix [NEQ]
    N_Vector pp;            /* vector of prec. diag. elements */
    int animationIndx;      //integer for storing filenames at different time steps for animation purposes [-]
    double mu_max;          //variable for storing max value of the chemical potential during simulation [-]
    double mu_min;          //variable for storing min value of the chemical potential during simulation [-]
    double rtol;            //variable for storing relative tolerance [-]
    int NOUT;               //number of iterations [-]
    int pos_RES_simTime;    //position of the simulation time in the results array [-]
    int pos_RES_Iapp;       //position of the applied current density in the results array [-]
    int pos_RES_dPhi;       //position of the voltage difference in the results array [-]
    int resultIndx;         //variable for storing current index for storing time dependent results [-]

} *data_spinodal_decomposition;


/* Prototypes of functions called by IDA */

int resrob(realtype tres, N_Vector yy, N_Vector yp,
           N_Vector resval, void *user_data);

int jacrob(realtype tt,  realtype cj,
           N_Vector yy, N_Vector yp, N_Vector resvec,
           SUNMatrix JJ, void *user_data,
           N_Vector tempv1, N_Vector tempv2, N_Vector tempv3);

/* Prototypes of private functions */
static void PrintHeader(realtype rtol, N_Vector avtol, N_Vector y);
static void PrintOutput(void *mem, realtype t, N_Vector y);
static void PrintFinalStats(void *mem);
static int check_retval(void *returnvalue, const char *funcname, int opt);

void dsd_allocate (data_spinodal_decomposition DSD);
void dsd_deallocate (data_spinodal_decomposition DSD);
void dsd_initialize(data_spinodal_decomposition DSD);
void dsd_set_initial_values(data_spinodal_decomposition DSD);
void calculate_CV_master_indeces (data_spinodal_decomposition DSD);
void dsd_set_properties (data_spinodal_decomposition DSD, int argc, char *argv[]);
void update_local_state_vector (N_Vector yy, data_spinodal_decomposition DSD);
void calculate_local_jacobian_matrix(realtype cj, data_spinodal_decomposition DSD);
void initialize_master_IDA_vectors (N_Vector yy, N_Vector yp, N_Vector avtol, N_Vector id, data_spinodal_decomposition DSD);
void clear_output_files(data_spinodal_decomposition DSD);
void write_jacobi_matrix (SUNMatrix JJ, data_spinodal_decomposition DSD);
void write_state_variable_matrix_form (data_spinodal_decomposition DSD, int pos_SV_write, int timeStepIndx);
void update_time_step (data_spinodal_decomposition DSD, long int numStep_old, long int numStep_new);
void write_params_and_sim_stats(void *mem, data_spinodal_decomposition DSD);
void write_species_conservation (data_spinodal_decomposition DSD);
double convert_ndn_to_real_time (data_spinodal_decomposition DSD, double ndn_time);
double convert_real_time_to_ndn (data_spinodal_decomposition DSD, double real_time);
void set_paths_output_files (data_spinodal_decomposition DSD);
int randomRangeInt (int minVal, int maxVal);
double randomRangeDouble (double minVal, double maxVal);
void calculate_stationary_state_level_chemical_potential(data_spinodal_decomposition DSD);
void write_stationary_level_mu (data_spinodal_decomposition DSD);
void write_restart_file (data_spinodal_decomposition DSD);
void read_restart_file_and_initialize_state_vector (data_spinodal_decomposition DSD);
void calculate_source_term (data_spinodal_decomposition DSD);
void calculate_average_ndn_domain_concentration_chemPotential (data_spinodal_decomposition DSD);
void generate_Li_source_pulses (data_spinodal_decomposition DSD);
double convert_real_Li_source_to_ndn (data_spinodal_decomposition DSD, double real_S);
void read_input_current_profile (data_spinodal_decomposition DSD);
void event_handler_input_data (data_spinodal_decomposition DSD);
void write_Crate_profile (data_spinodal_decomposition DSD);
double convert_Crate_per_second_to_ndn (data_spinodal_decomposition DSD, double Crate_per_second);
double convert_Crate_unit_per_hour_to_per_second (double Crate_per_hour);
double convert_Crate_per_second_to_ndn_Li_source (data_spinodal_decomposition DSD, double Crate_per_second);
double convert_Crate_unit_per_second_to_per_hour (double Crate_per_second);
void event_handler_pulse_generator (data_spinodal_decomposition DSD);
void calc_source_term_double_layer(N_Vector yp, data_spinodal_decomposition DSD);
void calc_jac_double_layer(realtype cj, data_spinodal_decomposition DSD);
double convert_ndn_to_real_dPhi (data_spinodal_decomposition DSD, double ndn_dPhi);
double calculate_derivative_BV_exponent_over_eta (data_spinodal_decomposition DSD, double ndn_eta_value);
double calculate_BV_exponent (data_spinodal_decomposition DSD, double ndn_eta_value);
void calculate_diagonal_Jac_elements (realtype cj, data_spinodal_decomposition DSD);
int PsetupSD(realtype tt, N_Vector yy, N_Vector yp, N_Vector rr, realtype c_j, void *user_data);
int PsolveSD(realtype tt, N_Vector yy, N_Vector yp, N_Vector rr, N_Vector rvec, N_Vector zvec, realtype cj, realtype delta, void *user_data);
void write_potential_difference (data_spinodal_decomposition DSD);
void calculate_CV_xy_coordinates (data_spinodal_decomposition DSD);
void write_animation_results (data_spinodal_decomposition DSD);
void set_paths_output_animation_files (data_spinodal_decomposition DSD);
void update_relative_absolute_tolerances (N_Vector avtol, data_spinodal_decomposition DSD);
void write_average_chemical_potential (data_spinodal_decomposition DSD);
int run_single_SD_simulation (data_spinodal_decomposition DSD, int argc, char *argv[]);
void update_simulation_parameters (data_spinodal_decomposition DSD, double dt, double Nout);
void store_time_dependent_results (data_spinodal_decomposition DSD);
void allocate_path_filenames(data_spinodal_decomposition DSD);
void deallocate_path_filenames(data_spinodal_decomposition DSD);
void read_restart_animation_index (data_spinodal_decomposition DSD);
void write_clock_timestep (data_spinodal_decomposition DSD);

/*
 *--------------------------------------------------------------------
 * Main Program
 *--------------------------------------------------------------------
 */

int main(int argc, char *argv[])
{
    int retval = 0; //return value

    //declare and initialize data spinodal decomposition structure
    data_spinodal_decomposition DSD = NULL;
    DSD = (data_spinodal_decomposition) malloc (sizeof *DSD);
    allocate_path_filenames(DSD);

    /* initialize random number generator */
    if(USE_constant_initial_C_mu == 0) srand(time(NULL));

    /* ----- single simulation run ----- */
    if (USE_run_mode == 0){

        update_simulation_parameters(DSD, dt_, NOUT_);
        retval = run_single_SD_simulation(DSD, argc, argv);
    } 


    deallocate_path_filenames(DSD);
    free(DSD); //free data spinodal decomposition structure


    return(retval);

}

/*
 *--------------------------------------------------------------------
 * Single spinodal decomposition simulation
 *--------------------------------------------------------------------
 */

 int run_single_SD_simulation (data_spinodal_decomposition DSD, int argc, char *argv[]){

    void *mem;
    N_Vector yy, yp, avtol, id;
    realtype rtol;
    realtype t0, tout1, tout, tret;
    int iout, retval, nRestarts = 0;
    SUNMatrix A;
    SUNLinearSolver LS;
    SUNNonlinearSolver NLS;
    long int numStep_old = 0;
    long int numStep_new = 0;
    clock_t begin, end;       //clock type of variables for start and end
    clock_t begin_timestep, end_timestep;       //clock type of variables to clock the individual timestep

    dsd_allocate(DSD);    //allocate data for diffusion equation
    dsd_initialize(DSD);  //initialize structure members of the data_spinodal_decomposition struct
    dsd_set_properties(DSD, argc, argv);  //set properties of the structure members
    dsd_set_initial_values(DSD);  //set initial values to the structure members of the data_spinodal_decomposition struct
    calculate_CV_master_indeces(DSD); //calculate index positions of the control volumes and their respective neighbours
    set_paths_output_files(DSD);//set paths for output files
    calculate_CV_xy_coordinates(DSD); //calculate x and y centroid coordinates of each control volume

    if (USE_restart_simulation == 1){
        read_restart_file_and_initialize_state_vector(DSD); //re-initialize state vector with data from the restart file
        read_restart_animation_index(DSD); //read and store last animation index
        iout = -1;  //set iout to -1 to prevent writing to file in the fist iteration
    }else {
        clear_output_files(DSD); //clear output files
        iout = 0;
    }

    //read C-rate profile from external file if selected
    if (USE_pulse_shape == 3) read_input_current_profile(DSD);

    //set start time of the first event
    if (USE_pulse_shape == 4){ ///TODO XXXXX
        DSD->PRP.eventTime = 0.2;
        DSD->PRP.ndn_eventTime = convert_real_time_to_ndn(DSD, DSD->PRP.eventTime);
    }


    mem = NULL;
    yy = yp = avtol = id = NULL;
    A = NULL;
    LS = NULL;
    NLS = NULL;
    DSD->pp = NULL;

    /* Allocate N-vectors. */
    yy = N_VNew_Serial(NEQ);
    if(check_retval((void *)yy, "N_VNew_Serial", 0)) return(1);
    yp = N_VNew_Serial(NEQ);
    if(check_retval((void *)yp, "N_VNew_Serial", 0)) return(1);
    avtol = N_VNew_Serial(NEQ);
    if(check_retval((void *)avtol, "N_VNew_Serial", 0)) return(1);
    id = N_VNew_Serial(NEQ);
    if(check_retval((void *)id, "N_VNew_Serial", 0)) return(1);
    DSD->pp = N_VNew_Serial(NEQ);
    if(check_retval((void *)DSD->pp, "N_VNew_Serial", 0)) return(1);

    /* Create and initialize  y, y', and absolute tolerance vectors. */
    rtol = RCONST(RTOL);
    initialize_master_IDA_vectors(yy,yp,avtol,id,DSD);

    /* Integration limits */
    t0 = DSD->ndn_simTime;
    tout1 = DSD->ndn_simTime + RCONST(DSD->PRP.ndn_dt);

    if (USE_run_mode == 0) PrintHeader(rtol, avtol, yy);

    /* Call IDACreate and IDAInit to initialize IDA memory */
    mem = IDACreate();
    if(check_retval((void *)mem, "IDACreate", 0)) return(1);

    /* Set user defined data*/
    retval = IDASetUserData(mem, DSD);
    if(check_retval(&retval, "IDASetUserData", 1)) return(1);

    /* Set which components are algebraic or differential */
    retval = IDASetId(mem, id);
    if(check_retval(&retval, "IDASetId", 1)) return(1);

    retval = IDAInit(mem, resrob, t0, yy, yp);
    if(check_retval(&retval, "IDAInit", 1)) return(1);

    /* Call IDASVtolerances to set tolerances */
    retval = IDASVtolerances(mem, DSD->rtol, avtol);
    if(check_retval(&retval, "IDASVtolerances", 1)) return(1);

    if (USE_solver_type == USE_DENSE){
        /* Create dense SUNMatrix for use in linear solves */
        A = SUNDenseMatrix(NEQ, NEQ);
        if(check_retval((void *)A, "SUNDenseMatrix", 0)) return(1);

        /* Create dense SUNLinearSolver object */
        LS = SUNLinSol_Dense(yy, A);
        if(check_retval((void *)LS, "SUNLinSol_Dense", 0)) return(1);

        /* Attach the matrix and linear solver */
        retval = IDASetLinearSolver(mem, LS, A);
        if(check_retval(&retval, "IDASetLinearSolver", 1)) return(1);

        /* Set the user-supplied Jacobian routine */
        retval = IDASetJacFn(mem, jacrob);
        if(check_retval(&retval, "IDASetJacFn", 1)) return(1);

        /* Create dense SUNNonLinearSolver object */
        NLS = SUNNonlinSol_Newton(yy);
        if(check_retval((void *)NLS, "SUNNonlinSol_Newton", 0)) return(1);

        /* Attach the nonlinear solver */
        retval = IDASetNonlinearSolver(mem, NLS);
        if(check_retval(&retval, "IDASetNonlinearSolver", 1)) return(1);

    } else if (USE_solver_type == USE_SPGMR){
        /* Create the linear solver SUNLinSol_SPGMR with left preconditioning
        and the default Krylov dimension */
        LS = SUNLinSol_SPGMR(yy, PREC_LEFT, 0);
        if(check_retval((void *)LS, "SUNLinSol_SPGMR", 0)) return(1);

    } else if (USE_solver_type == USE_SPFGMR){
        /* Create the linear solver SUNLinSol_SPFGMR with left preconditioning
        and the default Krylov dimension */
        LS = SUNLinSol_SPFGMR(yy, PREC_LEFT, 0);
        if(check_retval((void *)LS, "SUNLinSol_SPFGMR", 0)) return(1);

    } else if (USE_solver_type == USE_SPBCGS){
        /* Create the linear solver SUNLinSol_SPBCGS with left preconditioning
        and the default Krylov dimension */
        LS = SUNLinSol_SPBCGS(yy, PREC_LEFT, 0);
        if(check_retval((void *)LS, "SUNLinSol_SPBCGS", 0)) return(1);

    } else if (USE_solver_type == USE_SPTFQMR){
        /* Create the linear solver SUNLinSol_SPTFQMR with left preconditioning
        and the default Krylov dimension */
        LS = SUNLinSol_SPTFQMR(yy, PREC_LEFT, 0);
        if(check_retval((void *)LS, "SUNLinSol_SPTFQMR", 0)) return(1);

    } else if (USE_solver_type == USE_PCG){
        /* Create the linear solver SUNLinSol_PCG with left preconditioning
        and the default Krylov dimension */
        LS = SUNLinSol_PCG(yy, PREC_LEFT, 0);
        if(check_retval((void *)LS, "SUNLinSol_PCG", 0)) return(1);
    }

    //common setting for Krylov methods
    if (USE_solver_type == USE_SPGMR ||
      USE_solver_type == USE_SPFGMR ||
      USE_solver_type == USE_SPBCGS ||
      USE_solver_type == USE_SPTFQMR ||
      USE_solver_type == USE_PCG) {

        /* IDA recommends allowing up to 5 restarts (default is 0) */
        retval = SUNLinSol_SPGMRSetMaxRestarts(LS, 5);
        if(check_retval(&retval, "SUNLinSol_SPGMRSetMaxRestarts", 1)) return(1);

        /* Attach the linear solver */
        retval = IDASetLinearSolver(mem, LS, NULL);
        if(check_retval(&retval, "IDASetLinearSolver", 1)) return(1);

        /* Set the preconditioner solve and setup functions */
        retval = IDASetPreconditioner(mem, PsetupSD, PsolveSD);
        if(check_retval(&retval, "IDASetPreconditioner", 1)) return(1);

    }

    if (USE_IVC_calculation == 1){
        /* Call IDACalcIC to correct the initial values. */
        retval = IDACalcIC(mem, IDA_YA_YDP_INIT, tout1);
        if(check_retval(&retval, "IDACalcIC", 1)) return(1);

        /* Call IDAGetConsistentIC to return the corrected initial conditions calculated by IDACalcIC. */
        retval = IDAGetConsistentIC(mem, yy, yp);
        if(check_retval(&retval, "IDAGetConsistentIC", 1)) return(1);

        //update local state vector on the control volume level with the new values
        update_local_state_vector(yy,DSD);
    }


    /* In loop, call IDASolve, print results, and test for error.
     Break out of loop when NOUT preset output times have been reached. */

    begin = clock();          /*  ---start clock---  */
    tout = tout1;
    while(1) {

        if (USE_run_mode == 0){
            //print data to txt file every N steps only in single run mode
            if(iout%writeStep_species_conservation == 0 && iout>-1) write_species_conservation(DSD);
            if(iout%writeStep_average_mu == 0 && iout>-1) write_average_chemical_potential(DSD);
            if(iout%writeStep_stationary_level_mu == 0 && iout>-1) write_stationary_level_mu(DSD);
            if(iout%writeStep_restart_file == 0 && iout>-1) write_restart_file(DSD);
            if(iout%writeStep_Crate_profile == 0 && iout>-1) write_Crate_profile(DSD);
            if(iout%writeStep_potential_difference == 0 && iout>-1) write_potential_difference(DSD);
            if(iout%writeStep_clock_timestep == 0 && iout>-1) write_clock_timestep(DSD);

            //print data for animation purposes every N steps
            if (USE_write_animation_files == 1){

                if(iout%writeStep_animation_files == 0 && iout>-1) {
                    set_paths_output_animation_files(DSD);  //set paths for storing animation files
                    write_animation_results(DSD);
                    DSD->animationIndx++; //increase animation index
                }
            }
        }


        //print state variable in the matrix form at the start and at the end of the simulation
        if (USE_restart_simulation == 1){
            if(iout == DSD->NOUT-1) write_state_variable_matrix_form(DSD, DSD->pos_SV_C, iout); //at simulation restart print only end result
        } else {
            if(iout == 0 || iout == DSD->NOUT-1) write_state_variable_matrix_form(DSD, DSD->pos_SV_C, iout);
        }

        //update relative and absolute tolerances after few timesteps in order to help the simulation to start
        if (USE_update_tols_after_n_dts == 1 && iout == N_dt_update_tol){
            update_relative_absolute_tolerances(avtol, DSD);
            retval = IDAReInit(mem, DSD->ndn_simTime, yy, yp);
            retval = IDASVtolerances(mem, DSD->rtol, avtol);
        }

        //store data to the results structure
        if(USE_run_mode == 1) store_time_dependent_results(DSD);

	    begin_timestep = clock();                                                                                   /*  ---start timestep clock---  */
        retval = IDASolve(mem, tout, &tret, yy, yp, IDA_NORMAL);
        end_timestep = clock(); DSD->clock_timestep =  (double)(end_timestep - begin_timestep) / CLOCKS_PER_SEC;    /*  ---end timestep clock---  */

        //PrintOutput(mem,tret,yy);
        calculate_average_ndn_domain_concentration_chemPotential(DSD); //calculate average non-dimensional concentration and chemical potential in the domain

        DSD->ndn_simTime = tret; //store current time to the user data structure

        //update timestep
        if (USE_variable_timestep == 1){
            IDAGetNumSteps(mem, &numStep_new);
            update_time_step (DSD, numStep_old, numStep_new);
            numStep_old = numStep_new;
        }

        //check if retval is equal to -4 (convergence error) and restart simulation with the initial timestep and last saved state vector
        if (retval == -4 && nRestarts<N_MAX_restarts){
            read_restart_file_and_initialize_state_vector(DSD); //re-initialize state vector with data from the restart file
            initialize_master_IDA_vectors(yy,yp,avtol,id,DSD); //re-initialize master IDA vectors
            DSD->PRP.dt = dt_; //reset time step
            DSD->PRP.ndn_dt = convert_real_time_to_ndn (DSD, DSD->PRP.dt); //reset non-dimensional time step
            tout = DSD->ndn_simTime + RCONST(DSD->PRP.ndn_dt*0.1); //first time step is set to be 10x lower than dt

            /* Re-initialize IDA */
            retval = IDAReInit(mem, DSD->ndn_simTime, yy, yp);
            if (check_retval(&retval, "IDAReInit", 1)) return(1);

            nRestarts++; //increase restart counter
        }

        if(check_retval(&retval, "IDASolve", 1)){
            /* Free memory if simulation fails*/
            IDAFree(&mem);
            SUNNonlinSolFree(NLS);
            SUNLinSolFree(LS);
            SUNMatDestroy(A);
            N_VDestroy(avtol);
            N_VDestroy(yy);
            N_VDestroy(yp);
            N_VDestroy(id);
            N_VDestroy(DSD->pp);

            dsd_deallocate(DSD);

           return(1);
        }

        //generate pulses of Li source, if selected
        generate_Li_source_pulses(DSD);

        //calculate stationary level value for chemical potential
        if (iout%step_calculate_stationary_level_mu == 0) calculate_stationary_state_level_chemical_potential(DSD);
        if (DSD->stationary_level_mu < stationary_level_mu_MAX){ //end simulation if stationary level is achieved

            write_state_variable_matrix_form(DSD, DSD->pos_SV_C, DSD->NOUT-1);
            printf("\n========== Convergence criteria achieved in step %i of %i. ==========\n", iout, DSD->NOUT); //print console message
            break;
        }

        //increase time
        if (retval == IDA_SUCCESS) {
          iout++;
          tout += RCONST(DSD->PRP.ndn_dt);
        }

        //stop after NOUT iterations
        if (iout == DSD->NOUT) break;
    }

    end = clock(); DSD->clock_solver =  (double)(end - begin) / CLOCKS_PER_SEC;     /*  ---end clock---  */

    if (USE_run_mode == 0){
        PrintOutput(mem,tret,yy);
        PrintFinalStats(mem);

        write_params_and_sim_stats(mem, DSD); //write simulation parameters, settings and statistics to file
        write_restart_file(DSD); //write final restart file at the end of the simulation
    }


    /* Free memory */
    IDAFree(&mem);
    SUNNonlinSolFree(NLS);
    SUNLinSolFree(LS);
    SUNMatDestroy(A);
    N_VDestroy(avtol);
    N_VDestroy(yy);
    N_VDestroy(yp);
    N_VDestroy(id);
    N_VDestroy(DSD->pp);

    dsd_deallocate(DSD);    //deallocate data for spinodal decomposition simulation

    return(retval);
 }


/*
 *--------------------------------------------------------------------
 * Functions called by IDA
 *--------------------------------------------------------------------
 */

/*
 * Define the system residual function.
 */

int resrob(realtype tres, N_Vector yy, N_Vector yp, N_Vector rr, void *user_data)
{
  realtype *ypval;

  ypval = N_VGetArrayPointer(yp);

  data_spinodal_decomposition DSD;
  DSD = (data_spinodal_decomposition)user_data;

  int i_Ncv = N_CV_x*N_CV_y;      //total number of control volumes
  int i_N_var_tot = N_var_tot;    //number of state variables in each control volume

  double CP, CE, CW, CN, CS;       //variables for storing values of concentration in central and neighbouring control volumes
  double muP, muE, muW, muN, muS;  //variables for storing values of chemical potential in central and neighbouring control volumes
  double SP;                        //variable for storing source of Li in central control volume

  int C_P_RHS = 0;  //variable for storing position of RHS_fun(concentration) in central CV to master RHS vector
  int mu_P_RHS = 0;  //variable for storing position of RHS_fun(chemical potential) in central CV to master RHS vector
  int dPhi_RHS = i_Ncv * i_N_var_tot; //variable for storing position of RHS_fun(voltage difference dPhi) to master RHS vector

  int sv_C = DSD->pos_SV_C; //integer position of the concentration in the local state vector
  int sv_mu = DSD->pos_SV_mu; //integer position of the chemical potential in the local state vector
  int sv_dPhi = DSD->pos_LSC_dPhi; //integer position of the voltage difference in the local state vector

  double const Chixx_by_dx2 = DSD->PRP.ndn_Chi_xx / (DSD->PRP.ndn_dx*DSD->PRP.ndn_dx); //pre-calculated Chixx/(dx^2) for computational efficiency
  double const Chiyy_by_dy2 = DSD->PRP.ndn_Chi_yy / (DSD->PRP.ndn_dy*DSD->PRP.ndn_dy); //pre-calculated Chiyy/(dy^2) for computational efficiency
  double const kappaxx_by_dx2 = DSD->PRP.ndn_Kappa_xx / (DSD->PRP.ndn_dx*DSD->PRP.ndn_dx); //pre-calculated kappa_xx/(dx^2) for computational efficiency
  double const kappayy_by_dy2 = DSD->PRP.ndn_Kappa_yy / (DSD->PRP.ndn_dy*DSD->PRP.ndn_dy); //pre-calculated kappa_yy/(dy^2) for computational efficiency

  DSD->master_RHS = N_VGetArrayPointer(rr);

  //update local state vector on the control volume level with the new values
  update_local_state_vector(yy,DSD);

  //calculate source of Li to each control volume
  calculate_source_term(DSD);

  //calculate double layer source term
  if (USE_double_layer_model == 1){
    calc_source_term_double_layer(yp, DSD);
  }

  int i_CV = 0; //control volume counter

  //loop across CVs
  for (i_CV=0; i_CV<i_Ncv; i_CV++){

    CP = DSD->CV[DSD->CV[i_CV].pos_CV_P].sv_CV[sv_C];
    CE = DSD->CV[DSD->CV[i_CV].pos_CV_E].sv_CV[sv_C];
    CW = DSD->CV[DSD->CV[i_CV].pos_CV_W].sv_CV[sv_C];
    CN = DSD->CV[DSD->CV[i_CV].pos_CV_N].sv_CV[sv_C];
    CS = DSD->CV[DSD->CV[i_CV].pos_CV_S].sv_CV[sv_C];

    muP = DSD->CV[DSD->CV[i_CV].pos_CV_P].sv_CV[sv_mu];
    muE = DSD->CV[DSD->CV[i_CV].pos_CV_E].sv_CV[sv_mu];
    muW = DSD->CV[DSD->CV[i_CV].pos_CV_W].sv_CV[sv_mu];
    muN = DSD->CV[DSD->CV[i_CV].pos_CV_N].sv_CV[sv_mu];
    muS = DSD->CV[DSD->CV[i_CV].pos_CV_S].sv_CV[sv_mu];

    SP = DSD->CV[DSD->CV[i_CV].pos_CV_P].S_ndn;

    C_P_RHS = DSD->CV[i_CV].pos_CV_P * i_N_var_tot + DSD->pos_SV_C;
    mu_P_RHS = DSD->CV[i_CV].pos_CV_P * i_N_var_tot + DSD->pos_SV_mu;

    DSD->master_RHS[C_P_RHS] = Chixx_by_dx2 * ((CE*CE*(1.0-CE) + CP*CP*(1.0-CP)) * (muE-muP) -
                                               (CP*CP*(1.0-CP) + CW*CW*(1.0-CW)) * (muP-muW)) +

                               Chiyy_by_dy2 * ((CN*CN*(1.0-CN) + CP*CP*(1.0-CP)) * (muN-muP) -
                                               (CP*CP*(1.0-CP) + CS*CS*(1.0-CS)) * (muP-muS)) + SP - ypval[C_P_RHS];


    DSD->master_RHS[mu_P_RHS] = muP - log(CP/(1.0 - CP)) -
                            DSD->PRP.ndn_Omega * (1.0 - 2.0*CP) +
                            kappaxx_by_dx2 * (CE - 2.0*CP + CW) +
                            kappayy_by_dy2 * (CN - 2.0*CP + CS) -
                            DSD->PRP.ndn_B_strain * (CP - DSD->ndn_avg_C);

  }

  //lithium source control
  if (USE_source_term > 0){ //avoid memory leaks
    DSD->master_RHS[dPhi_RHS + sv_dPhi] = DSD->LSC.sum_S_ndn_CV - DSD->DLM.S_DL_ndn - DSD->PRP.ndn_S_TOT;
  }


  return(0);
}

/*
 * Define the Jacobian function.
 */

int jacrob(realtype tt,  realtype cj,
           N_Vector yy, N_Vector yp, N_Vector resvec,
           SUNMatrix JJ, void *user_data,
           N_Vector tempv1, N_Vector tempv2, N_Vector tempv3)
{

    data_spinodal_decomposition DSD;
    DSD = (data_spinodal_decomposition)user_data;

    //calculate jacobi contributions to double layer source term
    if (USE_double_layer_model == 1){
        calc_jac_double_layer(cj, DSD);
    }

    //calculate partial derivatives to local jacobi matrix
    calculate_local_jacobian_matrix(cj, DSD);

    int i_Ncv = N_CV_x*N_CV_y;      //total number of control volumes
    int i_N_var_tot = N_var_tot;    //number of state variables in each control volume

    int j,k; //counter

    //integer position of the partial derivative in local jacobian matrix of rhs function over variables in the central/east/west/north/south control volume [-]
    int pos_P = DSD->pos_jac_P;
    int pos_E = DSD->pos_jac_E;
    int pos_W = DSD->pos_jac_W;
    int pos_N = DSD->pos_jac_N;
    int pos_S = DSD->pos_jac_S;
    int pos_dPhi = DSD->pos_jac_dPhi;

    int sv_C = DSD->pos_SV_C;           //integer position of the concentration in the local state vector
    int sv_dPhi = DSD->pos_LSC_dPhi;    //integer position of the voltage difference in the local state vector
    int dPhi_JAC = i_Ncv * i_N_var_tot; //variable for storing position of lithium source control equation in JAC matrix

    int der_C = DSD->pos_deriv_C;       //integer position of the derivative over concentration
    int der_mu = DSD->pos_deriv_mu;     //integer position of the derivative over chemical potential

    int C_P = 0;    //variable for storing position of first state vector variable of central CV in the master rhs vector
    int C_E = 0;    //variable for storing position of first state vector variable of east CV in the master rhs vector
    int C_W = 0;    //variable for storing position of first state vector variable of west CV in the master rhs vector
    int C_N = 0;    //variable for storing position of first state vector variable of north CV in the master rhs vector
    int C_S = 0;    //variable for storing position of first state vector variable of south CV in the master rhs vector

    int i_CV = 0;   //control volume counter

    //loop across CVs
    for (i_CV=0; i_CV<i_Ncv; i_CV++){
        C_P = DSD->CV[i_CV].pos_CV_P * i_N_var_tot + DSD->pos_SV_C;
        C_E = DSD->CV[i_CV].pos_CV_E * i_N_var_tot + DSD->pos_SV_C;
        C_W = DSD->CV[i_CV].pos_CV_W * i_N_var_tot + DSD->pos_SV_C;
        C_N = DSD->CV[i_CV].pos_CV_N * i_N_var_tot + DSD->pos_SV_C;
        C_S = DSD->CV[i_CV].pos_CV_S * i_N_var_tot + DSD->pos_SV_C;

        for (j=0; j<i_N_var_tot; j++){
            for (k=0; k<i_N_var_tot; k++){
                IJth(JJ,C_P+j,C_P+k) = DSD->CV[i_CV].jac_CV[j][k][pos_P];
                IJth(JJ,C_P+j,C_E+k) = DSD->CV[i_CV].jac_CV[j][k][pos_E];
                IJth(JJ,C_P+j,C_W+k) = DSD->CV[i_CV].jac_CV[j][k][pos_W];
                IJth(JJ,C_P+j,C_N+k) = DSD->CV[i_CV].jac_CV[j][k][pos_N];
                IJth(JJ,C_P+j,C_S+k) = DSD->CV[i_CV].jac_CV[j][k][pos_S];
            }
        }

        //lithium source control - control volume related dependencies
        if (USE_source_term > 0){ //avoid memory leaks
            ///dFC/ddPhi
            IJth(JJ,C_P,dPhi_JAC) =  DSD->CV[i_CV].jac_CV[sv_C][sv_C][pos_dPhi];

            ///dF_LSC/dC
            IJth(JJ,dPhi_JAC, C_P) = DSD->CV[i_CV].dS_ndn_dSVar[der_C];

            ///dF_LSC/dmu
            IJth(JJ,dPhi_JAC, C_P+DSD->pos_SV_mu) = DSD->CV[i_CV].dS_ndn_dSVar[der_mu];
        }
    }

    //lithium source control - equation itself
    if (USE_source_term > 0){ //avoid memory leaks
        ///dF_LSC/ddPhi
        IJth(JJ,dPhi_JAC,dPhi_JAC) =  DSD->LSC.jac_LSC[sv_dPhi][sv_dPhi][sv_dPhi];
    }

  return(0);
}

/*
 *--------------------------------------------------------------------
 * Private functions
 *--------------------------------------------------------------------
 */

/*
 * Print first lines of output (problem description)
 */

static void PrintHeader(realtype rtol, N_Vector avtol, N_Vector y)
{
  realtype *atval, *yval;

  atval = N_VGetArrayPointer(avtol);
  yval  = N_VGetArrayPointer(y);

  printf("\nSpinodal decomposition with Cahn-Hilliard 2D DAE serial problem for IDA developed by University of Ljubljana\n");
  printf("Grid size: %i x %i\n", N_CV_x, N_CV_y);
  if(USE_solver_type == USE_DENSE) printf("Linear solver: DENSE, with user-supplied Jacobian.\n");
  if(USE_solver_type == USE_SPGMR) printf("Linear solver: SPGMR, preconditioner using diagonal elements.\n");
  if(USE_solver_type == USE_SPFGMR) printf("Linear solver: SPFGMR, preconditioner using diagonal elements.\n");
  if(USE_solver_type == USE_SPBCGS) printf("Linear solver: SPBCGS, preconditioner using diagonal elements.\n");
  if(USE_solver_type == USE_SPTFQMR) printf("Linear solver: SPTFQMR, preconditioner using diagonal elements.\n");
  if(USE_solver_type == USE_PCG) printf("Linear solver: PCG, preconditioner using diagonal elements.\n");

#if defined(SUNDIALS_EXTENDED_PRECISION)
  printf("Tolerance parameters:  rtol = %Lg   atol = %Lg %Lg %Lg \n",
         rtol, atval[0],atval[1],atval[2]);
  printf("Initial conditions y0 = (%Lg %Lg %Lg)\n",
         yval[0], yval[1], yval[2]);
#elif defined(SUNDIALS_DOUBLE_PRECISION)
  printf("Tolerance parameters:  rtol = %g   atol = %g %g %g \n",
         rtol, atval[0],atval[1],atval[2]);
  printf("Initial conditions y0 = (%g %g %g)\n",
         yval[0], yval[1], yval[2]);
#else
  printf("Tolerance parameters:  rtol = %g   atol = %g %g %g \n",
         rtol, atval[0],atval[1],atval[2]);
  printf("Initial conditions y0 = (%g %g %g)\n",
         yval[0], yval[1], yval[2]);
#endif
  printf("Constraints and id not used.\n\n");
  printf("-----------------------------------------------------------------------\n");
  printf("  t             y1           y2           y3");
  printf("      | nst  k      h\n");
  printf("-----------------------------------------------------------------------\n");
}

/*
 * Print Output
 */

static void PrintOutput(void *mem, realtype t, N_Vector y)
{
  realtype *yval;
  int retval, kused;
  long int nst;
  realtype hused;

  yval  = N_VGetArrayPointer(y);

  retval = IDAGetLastOrder(mem, &kused);
  check_retval(&retval, "IDAGetLastOrder", 1);
  retval = IDAGetNumSteps(mem, &nst);
  check_retval(&retval, "IDAGetNumSteps", 1);
  retval = IDAGetLastStep(mem, &hused);
  check_retval(&retval, "IDAGetLastStep", 1);
#if defined(SUNDIALS_EXTENDED_PRECISION)
  printf("%10.4Le %12.4Le %12.4Le %12.4Le | %3ld  %1d %12.4Le\n",
         t, yval[0], yval[1], yval[2], nst, kused, hused);
#elif defined(SUNDIALS_DOUBLE_PRECISION)
  printf("%10.4e %12.4e %12.4e %12.4e | %3ld  %1d %12.4e\n",
         t, yval[0], yval[1], yval[2], nst, kused, hused);
#else
  printf("%10.8e %12.8e %12.8e %12.8e | %3ld  %1d %12.4e\n",
         t, yval[0], yval[1], yval[2], nst, kused, hused);
#endif
}


/*
 * Print final integrator statistics
 */

static void PrintFinalStats(void *mem)
{
  int retval;
  long int nst, nni, nje, nre, nreLS, netf, ncfn, nge;

  retval = IDAGetNumSteps(mem, &nst);
  check_retval(&retval, "IDAGetNumSteps", 1);
  retval = IDAGetNumResEvals(mem, &nre);
  check_retval(&retval, "IDAGetNumResEvals", 1);
  retval = IDAGetNumJacEvals(mem, &nje);
  check_retval(&retval, "IDAGetNumJacEvals", 1);
  retval = IDAGetNumNonlinSolvIters(mem, &nni);
  check_retval(&retval, "IDAGetNumNonlinSolvIters", 1);
  retval = IDAGetNumErrTestFails(mem, &netf);
  check_retval(&retval, "IDAGetNumErrTestFails", 1);
  retval = IDAGetNumNonlinSolvConvFails(mem, &ncfn);
  check_retval(&retval, "IDAGetNumNonlinSolvConvFails", 1);
  retval = IDAGetNumLinResEvals(mem, &nreLS);
  check_retval(&retval, "IDAGetNumLinResEvals", 1);
  retval = IDAGetNumGEvals(mem, &nge);
  check_retval(&retval, "IDAGetNumGEvals", 1);

  printf("\nFinal Run Statistics: \n\n");
  printf("Number of steps                    = %ld\n", nst);
  printf("Number of residual evaluations     = %ld\n", nre+nreLS);
  printf("Number of Jacobian evaluations     = %ld\n", nje);
  printf("Number of nonlinear iterations     = %ld\n", nni);
  printf("Number of error test failures      = %ld\n", netf);
  printf("Number of nonlinear conv. failures = %ld\n", ncfn);
  printf("Number of root fn. evaluations     = %ld\n", nge);
}

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns an integer value so check if
 *            retval < 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */

static int check_retval(void *returnvalue, const char *funcname, int opt)
{
  int *retval;
  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && returnvalue == NULL) {
    fprintf(stderr,
            "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1);
  } else if (opt == 1) {
    /* Check if retval < 0 */
    retval = (int *) returnvalue;
    if (*retval < 0) {
      fprintf(stderr,
              "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
              funcname, *retval);
      return(1);
    }
  } else if (opt == 2 && returnvalue == NULL) {
    /* Check if function returned NULL pointer - no memory allocated */
    fprintf(stderr,
            "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1);
  }

  return(0);
}


void dsd_allocate (data_spinodal_decomposition DSD){
    int i, j, k;                        //counters
    int i_Ncv = N_CV_x*N_CV_y;          //total number of control volumes
    int i_N_var_tot = N_var_tot;        //number of state variables in each control volume
    int i_N_var_LSC_tot = N_var_LSC_tot;//number of active state variables in lithium source control equation
    int i_N_var_derivs = N_var_derivs;  //total number of variables to calculate partial derivatives over
    int i_CV_pos = N_CV_pos;            //number of control volume positions [2D = P E W N S] and additional dependent variables
    int i_LSC_pos = N_LSC_pos;          //number of control volume positions for the lithium source control equation
    int i_NEQ = NEQ;                    //dimension of the problem

    //main structure
    DSD->resultPath = malloc(Len_filename * sizeof(char));
    DSD->animationResultPath = malloc(Len_filename * sizeof(char));
    DSD->JacDiag = malloc(i_NEQ * sizeof(double));

    //control volume structure
    DSD->CV = malloc(sizeof(control_volume) * i_Ncv);
    for (i=0; i<i_Ncv; i++){
        DSD->CV[i].sv_CV = malloc(sizeof(double) * i_N_var_tot);
        DSD->CV[i].atval_CV = malloc(sizeof(double) * i_N_var_tot);
        DSD->CV[i].idval_CV= malloc(sizeof(double) * i_N_var_tot);
        DSD->CV[i].dS_ndn_dSVar = malloc(sizeof(double) * i_N_var_derivs);
        DSD->CV[i].jac_CV = malloc(sizeof(double**) * i_N_var_tot);

        for (j=0; j<i_N_var_tot; j++){
            DSD->CV[i].jac_CV[j] = malloc(sizeof(double*) * i_N_var_tot);

            for (k=0; k<i_N_var_tot; k++){
                DSD->CV[i].jac_CV[j][k] = malloc(sizeof(double) * i_CV_pos);
            }
        }
    }


    //lithium source control structure
    DSD->LSC.sv_LSC = malloc(sizeof(double) * i_N_var_LSC_tot);
    DSD->LSC.atval_LSC = malloc(sizeof(double) * i_N_var_LSC_tot);
    DSD->LSC.sum_dS_ndn_dSVar = malloc(sizeof(double) * i_N_var_derivs);
    DSD->LSC.jac_LSC = malloc(sizeof(double**) * i_N_var_LSC_tot);

    for (i=0; i<i_N_var_LSC_tot; i++){
        DSD->LSC.jac_LSC[i] = malloc(sizeof(double*) * i_N_var_LSC_tot);
        for (j=0; j<i_N_var_LSC_tot; j++){
            DSD->LSC.jac_LSC[i][j] = malloc(sizeof(double) * i_LSC_pos);
        }
    }

    //double layer model structure
    DSD->DLM.dS_DL_ndn_dSVar = malloc(sizeof(double) * i_N_var_derivs);

}

void allocate_path_filenames(data_spinodal_decomposition DSD){
    //path filename structure
    DSD->PFN.filename_jacobi = malloc(Len_filename * sizeof(char));
    DSD->PFN.filename_matrix_chem_potential_start = malloc(Len_filename * sizeof(char));
    DSD->PFN.filename_matrix_chem_potential_end = malloc(Len_filename * sizeof(char));
    DSD->PFN.filename_matrix_concentration_start = malloc(Len_filename * sizeof(char));
    DSD->PFN.filename_matrix_concentration_end = malloc(Len_filename * sizeof(char));
    DSD->PFN.filename_parameters_and_statistic = malloc(Len_filename * sizeof(char));
    DSD->PFN.filename_species_conservation = malloc(Len_filename * sizeof(char));
    DSD->PFN.filename_stationary_level_mu = malloc(Len_filename * sizeof(char));
    DSD->PFN.filename_restart_file = malloc(Len_filename * sizeof(char));
    DSD->PFN.filename_input_Crate_profile = malloc(Len_filename * sizeof(char));
    DSD->PFN.filename_output_Crate_profile = malloc(Len_filename * sizeof(char));
    DSD->PFN.filename_potential_difference = malloc(Len_filename * sizeof(char));
    DSD->PFN.filename_clock_timestep = malloc(Len_filename * sizeof(char));
    DSD->PFN.filename_animation_concentration = malloc(Len_filename * sizeof(char));
    DSD->PFN.filename_animation_chem_potential = malloc(Len_filename * sizeof(char));
    DSD->PFN.filename_animation_supporting_data = malloc(Len_filename * sizeof(char));
    DSD->PFN.filename_animation_time_data = malloc(Len_filename * sizeof(char));
    DSD->PFN.filename_average_chemical_potential = malloc(Len_filename * sizeof(char));
}


void dsd_deallocate (data_spinodal_decomposition DSD){
    int i,j,k;  //counter
    int i_Ncv = N_CV_x*N_CV_y;      //total number of control volumes
    int i_N_var_tot = N_var_tot;    //number of state variables in each control volume
    int i_N_var_LSC_tot = N_var_LSC_tot;//number of active state variables in lithium source control equation

    //control volume structure
    for (i=0; i<i_Ncv; i++){

        for (j=0; j<i_N_var_tot; j++){
            for (k=0; k<i_N_var_tot; k++){
                free(DSD->CV[i].jac_CV[j][k]);
            }
            free(DSD->CV[i].jac_CV[j]);
        }

        free(DSD->CV[i].sv_CV);
        free(DSD->CV[i].atval_CV);
        free(DSD->CV[i].idval_CV);
        free(DSD->CV[i].dS_ndn_dSVar);
        free(DSD->CV[i].jac_CV);
    }
    free(DSD->CV);


    //lithium source control structure
    for (i=0; i<i_N_var_LSC_tot; i++){
        for (j=0; j<i_N_var_LSC_tot; j++){
            free(DSD->LSC.jac_LSC[i][j]);
        }
        free(DSD->LSC.jac_LSC[i]);
    }

    free(DSD->LSC.sv_LSC);
    free(DSD->LSC.atval_LSC);
    free(DSD->LSC.jac_LSC);
    free(DSD->LSC.sum_dS_ndn_dSVar);

    //input data structure
    if (USE_pulse_shape == 3){//deallocate arrays if selected
        free(DSD->INDAT.time_data);
        free(DSD->INDAT.Crate_data);
    }

    //double layer model structure
    free(DSD->DLM.dS_DL_ndn_dSVar);

    //main structure
    free(DSD->resultPath);
    free(DSD->animationResultPath);
    free(DSD->JacDiag);

}

void deallocate_path_filenames(data_spinodal_decomposition DSD){
    //path filename structure
    free(DSD->PFN.filename_jacobi);
    free(DSD->PFN.filename_matrix_chem_potential_start);
    free(DSD->PFN.filename_matrix_chem_potential_end);
    free(DSD->PFN.filename_matrix_concentration_start);
    free(DSD->PFN.filename_matrix_concentration_end);
    free(DSD->PFN.filename_parameters_and_statistic);
    free(DSD->PFN.filename_species_conservation);
    free(DSD->PFN.filename_stationary_level_mu);
    free(DSD->PFN.filename_restart_file);
    free(DSD->PFN.filename_input_Crate_profile);
    free(DSD->PFN.filename_output_Crate_profile);
    free(DSD->PFN.filename_potential_difference);
    free(DSD->PFN.filename_clock_timestep);
    free(DSD->PFN.filename_animation_concentration);
    free(DSD->PFN.filename_animation_chem_potential);
    free(DSD->PFN.filename_animation_supporting_data);
    free(DSD->PFN.filename_animation_time_data);
    free(DSD->PFN.filename_average_chemical_potential);
}

void dsd_initialize(data_spinodal_decomposition DSD){
    int i,j,k,l;  //counter
    int i_Ncv = N_CV_x*N_CV_y;      //total number of control volumes
    int i_N_var_tot = N_var_tot;    //number of state variables in each control volume
    int i_N_var_LSC_tot = N_var_LSC_tot;//number of active state variables in lithium source control equation
    int i_N_var_derivs = N_var_derivs;  //total number of variables to calculate partial derivatives over
    int i_CV_pos = N_CV_pos;        //number of control volume positions [2D = P E W N S] and additional dependent variables
    int i_LSC_pos = N_LSC_pos;      //number of control volume positions for the lithium source control equation
    int i_NEQ = NEQ;                //dimension of the problem

    //main structure
    for (i=0; i<i_NEQ; i++){
        DSD->JacDiag[i] = 0.0;
    }

    //DSD structure
    DSD->pos_SV_C = 0;
    DSD->pos_SV_mu = 0;
    DSD->pos_LSC_dPhi = 0;
    DSD->pos_jac_P = 0;
    DSD->pos_jac_E = 0;
    DSD->pos_jac_W = 0;
    DSD->pos_jac_N = 0;
    DSD->pos_jac_S = 0;
    DSD->pos_jac_dPhi = 0;
    DSD->pos_deriv_C = 0;
    DSD->pos_deriv_dPhi = 0;
    DSD->pos_deriv_mu = 0;
    DSD->ndn_simTime = 0.0;
    DSD->clock_solver = 0.0;
    DSD->stationary_level_mu = 0.0;
    DSD->i_USE_source_term = 0;
    DSD->ndn_avg_C = 0.0;
    DSD->ndn_avg_mu = 0.0;
    DSD->pulse_relax_ID = 0;
    DSD->animationIndx = 0;
    DSD->mu_max = 0.0;
    DSD->mu_min = 0.0;
    DSD->rtol = 0.0;
    DSD->pos_RES_simTime = 0;
    DSD->pos_RES_Iapp = 0;
    DSD->pos_RES_dPhi = 0;
    DSD->resultIndx = 0;
    DSD->clock_timestep = 0.0;

    //properties structure
    //dimensional properties
    DSD->PRP.Lx = 0.0;
    DSD->PRP.dx = 0.0;
    DSD->PRP.Ly = 0.0;
    DSD->PRP.dy = 0.0;
    DSD->PRP.Omega = 0.0;
    DSD->PRP.C0 = 0.0;
    DSD->PRP.Cm = 0.0;
    DSD->PRP.D_xx = 0.0;
    DSD->PRP.D_yy = 0.0;
    DSD->PRP.Kappa_xx = 0.0;
    DSD->PRP.Kappa_yy = 0.0;
    DSD->PRP.rndCAmp = 0.0;
    DSD->PRP.dp = 0.0;
    DSD->PRP.i0 = 0.0;
    DSD->PRP.S0 = 0.0;
    DSD->PRP.alpha_a = 0.0;
    DSD->PRP.alpha_c = 0.0;
    DSD->PRP.S_TOT = 0.0;
    DSD->PRP.C_rate = 0.0;
    DSD->PRP.tmp_C_rate = 0.0;
    DSD->PRP.pulseTime = 0.0;
    DSD->PRP.relaxTime = 0.0;
    DSD->PRP.eventTime = 0.0;
    DSD->PRP.C_DL = 0.0;
    DSD->PRP.B_strain = 0.0;
    DSD->PRP.tau = 0.0;

    //non-dimensional properties
    DSD->PRP.ndn_Kappa_xx = 0.0;
    DSD->PRP.ndn_Kappa_yy = 0.0;
    DSD->PRP.ndn_Chi_xx = 0.0;
    DSD->PRP.ndn_Chi_yy = 0.0;
    DSD->PRP.ndn_Omega = 0.0;
    DSD->PRP.ndn_dx = 0.0;
    DSD->PRP.ndn_dy = 0.0;
    DSD->PRP.ndn_C0 = 0.0;
    DSD->PRP.ndn_rndCAmp = 0.0;
    DSD->PRP.ndn_S0 = 0.0;
    DSD->PRP.ndn_C_rate = 0.0;
    DSD->PRP.ndn_pulseTime = 0.0;
    DSD->PRP.ndn_relaxTime = 0.0;
    DSD->PRP.ndn_eventTime = 0.0;
    DSD->PRP.ndn_C_DL = 0.0;
    DSD->PRP.ndn_B_strain = 0.0;

    //control volume structure
    for (i=0; i<i_Ncv; i++){
        DSD->CV[i].pos_CV_E = 0;
        DSD->CV[i].pos_CV_P = 0;
        DSD->CV[i].pos_CV_W = 0;
        DSD->CV[i].pos_CV_N = 0;
        DSD->CV[i].pos_CV_S = 0;
        DSD->CV[i].S_ndn = 0.0;
        DSD->CV[i].xCV = 0.0;
        DSD->CV[i].ndn_xCV = 0.0;
        DSD->CV[i].yCV = 0.0;
        DSD->CV[i].ndn_yCV = 0.0;

        for (j=0; j<i_N_var_tot; j++){
            DSD->CV[i].sv_CV[j] = 0.0;
            DSD->CV[i].atval_CV[j] = 0.0;
            DSD->CV[i].idval_CV[j] = 0.0;

            for (k=0; k<i_N_var_tot; k++){
                for (l=0; l<i_CV_pos; l++){
                    DSD->CV[i].jac_CV[j][k][l] = 0.0;
                }

            }
        }

        for (j=0; j<i_N_var_derivs; j++){
            DSD->CV[i].dS_ndn_dSVar[j] = 0.0;
        }
    }

    //lithium source control structure
    DSD->LSC.sum_S_ndn_CV = 0.0;

    for (i=0; i<i_N_var_LSC_tot; i++){
        DSD->LSC.atval_LSC[i] = 0.0;
        DSD->LSC.sv_LSC[i] = 0.0;

        for (j=0; j<i_N_var_LSC_tot; j++){
            for (k=0; k<i_LSC_pos; k++){
                DSD->LSC.jac_LSC[i][j][k] = 0.0;
            }
        }
    }

    for (i=0; i<i_N_var_derivs; i++){
        DSD->LSC.sum_dS_ndn_dSVar[i] = 0.0;
    }

    //input data structure
    DSD->INDAT.dataIndx = 0;
    DSD->INDAT.dataLength = 0;
    DSD->INDAT.delaySwitch = 0;

    //double layer model structure
    for (i=0; i<i_N_var_derivs; i++){
        DSD->DLM.dS_DL_ndn_dSVar[i] = 0.0;
    }

}

void dsd_set_initial_values (data_spinodal_decomposition DSD){
    int i;  //counter
    int tmp = 0;  //counter
    int i_Ncv = N_CV_x*N_CV_y;      //total number of control volumes
    int i_randCV;                   //variable for storing random index of control volume
    int max_attempts = 1e6;         //maximum number of attempts in achieving maxDifference
    double sumC = 0.0;              //variable for summation of the concentration across all control volumes
    double sumMu = 0.0;             //variable for summation of the chemical potential across all control volumes
    double sumTheory = i_Ncv * DSD->PRP.ndn_C0; //theoretical sum of concentration
    double maxDifference = 1.0e-6;  //define maximal difference between sumC and sumTheory
    double randConc = 0.0;          //variable for storing new random value of concentration
    double diff_old = 0.0;          //temporary variable for old value of  difference between sumC and sumTheory
    double diff_new = 0.0;          //temporary variable for new value of difference between sumC and sumTheory

    //set positions in the state vector
    DSD->pos_SV_C = 0;
    DSD->pos_SV_mu = 1;

    //set positions in the LSC state vector
    DSD->pos_LSC_dPhi = 0;

    //set positions in the local jacobian matrix
    DSD->pos_jac_P = 0;
    DSD->pos_jac_E = 1;
    DSD->pos_jac_W = 2;
    DSD->pos_jac_N = 3;
    DSD->pos_jac_S = 4;
    DSD->pos_jac_dPhi = 5;

    //set position of the partial derivatives in the partial derivatives array
    DSD->pos_deriv_C = 0;
    DSD->pos_deriv_mu = 1;
    DSD->pos_deriv_dPhi = 2;

    //integer positions in the results array
    DSD->pos_RES_simTime = 0;
    DSD->pos_RES_Iapp = 1;
    DSD->pos_RES_dPhi = 2;

    //set selected source term ansatz
    DSD->i_USE_source_term = USE_source_term;

    //set initial ID for event handler: [=0]start with relaxation; [=2]start with pulse
    DSD->pulse_relax_ID = 2;


    //perform an initial calculation of the concentration and set the algebraic/differential components in the id vector
    for (i=0; i<i_Ncv; i++){
        DSD->CV[i].sv_CV[DSD->pos_SV_C] = DSD->PRP.ndn_C0 + randomRangeDouble(-DSD->PRP.ndn_rndCAmp, DSD->PRP.ndn_rndCAmp); //initial concentration [-]
        sumC += DSD->CV[i].sv_CV[DSD->pos_SV_C];

        DSD->CV[i].idval_CV[DSD->pos_SV_C] = ONE;
    }

    //initiate algorithm for bringing total species concentration closer to the theoretical one
    while (fabs(sumC-sumTheory) > maxDifference){
        //randomly select control volume index
        i_randCV = randomRangeInt(0, i_Ncv-1);

        //calculate new random value for concentration
        randConc = DSD->PRP.ndn_C0 + randomRangeDouble(-DSD->PRP.ndn_rndCAmp, DSD->PRP.ndn_rndCAmp);

        //calculate old and new differences
        diff_old = fabs(sumC - sumTheory);
        diff_new = fabs(sumC - DSD->CV[i_randCV].sv_CV[DSD->pos_SV_C] + randConc - sumTheory);

        //compare differences
        if (diff_new < diff_old) {
            //set new value for the sumC
            sumC -= DSD->CV[i_randCV].sv_CV[DSD->pos_SV_C];
            sumC += randConc;

            //set new value for the concentration
            DSD->CV[i_randCV].sv_CV[DSD->pos_SV_C] = randConc;

        }

        //increase counter
        tmp++;

        //store initial average value of non-dimensional concentration to the structure
        DSD->ndn_avg_C = sumC/i_Ncv;

        //stop if number of attempts exceeds max_attempts
        if (tmp>max_attempts){
            break;
        }
    }

    //based on concentration values calculate initial values also for chemical potential and set absolute tolerances for state vector variables
    for (i=0; i<i_Ncv; i++){
        DSD->CV[i].sv_CV[DSD->pos_SV_mu] = log(DSD->CV[i].sv_CV[DSD->pos_SV_C]/(1.0-DSD->CV[i].sv_CV[DSD->pos_SV_C])) +
                                            DSD->PRP.ndn_Omega * (1.0 - 2.0*DSD->CV[i].sv_CV[DSD->pos_SV_C]); //initial non-dimensional chemical potential []

        sumMu += DSD->CV[i].sv_CV[DSD->pos_SV_mu];

        if (USE_update_tols_after_n_dts == 0){
            DSD->CV[i].atval_CV[DSD->pos_SV_C] = ATOL_C; //absolute tolerance for non-dimensional concentration [-]
            DSD->CV[i].atval_CV[DSD->pos_SV_mu] = ATOL_mu; //absolute tolerance for non-dimensional chemical potential [-]

        } else if (USE_update_tols_after_n_dts == 1){
            //initial lowered absolute tolerance value to help simulation get started
            DSD->CV[i].atval_CV[DSD->pos_SV_C] = ATOL_C_INIT;
            DSD->CV[i].atval_CV[DSD->pos_SV_mu] = ATOL_mu_INIT;
        }

    }

    //store initial average value of non-dimensional chemical potential to the structure
    DSD->ndn_avg_mu = sumMu/i_Ncv;

    //set initial value of the potential difference
    DSD->LSC.sv_LSC[DSD->pos_LSC_dPhi] = -DSD->ndn_avg_mu;

    if (USE_update_tols_after_n_dts == 0){
        //absolute tolerance for non-dimensional voltage difference [-]
        DSD->LSC.atval_LSC[DSD->pos_LSC_dPhi] = ATOL_dPhi;

        //set relative tolerance
        DSD->rtol = RTOL;

    } else if (USE_update_tols_after_n_dts == 1){
        //initial lowered absolute tolerance value to help simulation get started
        DSD->LSC.atval_LSC[DSD->pos_LSC_dPhi] = ATOL_dPhi_INIT;

        //set relative tolerance
        DSD->rtol = RTOL_INIT;
    }


}


void dsd_set_properties (data_spinodal_decomposition DSD, int argc, char *argv[]){
    int i_Ncv_x = N_CV_x;      //number of control volumes in x-direction
    int i_Ncv_y = N_CV_y;      //number of control volumes in y-direction

    //dimensional properties
    DSD->PRP.Omega = Omega_;
    DSD->PRP.Cm = cm;
    DSD->PRP.Kappa_xx = Kappa_xx_;
    DSD->PRP.Kappa_yy = Kappa_yy_;
    DSD->PRP.dp = dp_;
    DSD->PRP.i0 = i0_;
    DSD->PRP.S0 = DSD->PRP.i0/(F_CONST * DSD->PRP.dp);
    DSD->PRP.alpha_a = alpha_a_;
    DSD->PRP.alpha_c = alpha_c_;
    DSD->PRP.tmp_C_rate = C_rate_;
    DSD->PRP.pulseTime = pulseTime_;
    DSD->PRP.relaxTime = relaxTime_;
    DSD->PRP.C_DL = C_DL_;
    DSD->PRP.B_strain = B_strain_;


    if (argc == 1){ //no command line arguments => use default values
        DSD->PRP.C0 = c0;
        DSD->PRP.rndCAmp = rndCAmp_;
        DSD->PRP.Lx = L_x;
        DSD->PRP.Ly = L_y;
        strcpy(DSD->resultPath, "./Results/default/");
        strcpy(DSD->PFN.filename_input_Crate_profile, "./InputData/default_Crate_profile.txt");

    } else if (argc == N_argc){ //use command line values
        sscanf(argv[1],"%lf", &DSD->PRP.C0);        //initial mean value for concentration [mol/m^3]
        sscanf(argv[2],"%lf", &DSD->PRP.rndCAmp);   //random amplitude of perturbation around initial dimensional concentration [mol/m^3]
        sscanf(argv[3],"%lf", &DSD->PRP.Lx);        //domain size in x-direction [m]
        sscanf(argv[3],"%lf", &DSD->PRP.Ly);        //domain size in y-direction [m]
        strcpy(DSD->resultPath, argv[4]);           //relative path to the selected folder for storing simulation results

    } else { //print error message and exit the program with code 1
        fprintf(stderr, "Error in using command line arguments: usage: %s [CO] [rndCAmp] [Lx==Ly] [./relative_path/]\n", argv[0]);
        exit(1);
    }

    DSD->PRP.dx = DSD->PRP.Lx / i_Ncv_x;
    DSD->PRP.dy = DSD->PRP.Ly / i_Ncv_y;
    DSD->PRP.D_xx = D_xx_;
    DSD->PRP.D_yy = D_yy_;
    DSD->PRP.tau = DSD->PRP.Lx*DSD->PRP.Lx/DSD->PRP.D_xx + DSD->PRP.Ly*DSD->PRP.Ly/DSD->PRP.D_yy;

    //set animation path by concatenating /Animation string to the result path
    snprintf(DSD->animationResultPath, Len_filename*sizeof(char), "%s%s", DSD->resultPath, "Animation/");


    if (USE_pulse_shape == 0) DSD->PRP.C_rate = convert_Crate_unit_per_hour_to_per_second(DSD->PRP.tmp_C_rate);
    DSD->PRP.S_TOT = DSD->PRP.C_rate * DSD->PRP.Cm * N_CV_x * N_CV_y;

    //non-dimensional properties
    DSD->PRP.ndn_dt     =   DSD->PRP.dt / DSD->PRP.tau;
    DSD->PRP.ndn_Chi_xx =   2.0*DSD->PRP.D_xx / (DSD->PRP.D_xx + DSD->PRP.D_yy) *
                            sqrt((DSD->PRP.Lx*DSD->PRP.Lx + DSD->PRP.Ly*DSD->PRP.Ly)/(2.0*DSD->PRP.Lx*DSD->PRP.Lx));
    DSD->PRP.ndn_Chi_yy =   2.0*DSD->PRP.D_yy / (DSD->PRP.D_xx + DSD->PRP.D_yy) *
                            sqrt((DSD->PRP.Lx*DSD->PRP.Lx + DSD->PRP.Ly*DSD->PRP.Ly)/(2.0*DSD->PRP.Ly*DSD->PRP.Ly));

    DSD->PRP.ndn_Kappa_xx = 2.0*DSD->PRP.Kappa_xx / (R*T*DSD->PRP.Cm*(DSD->PRP.Lx*DSD->PRP.Lx + DSD->PRP.Ly*DSD->PRP.Ly)) *
                            sqrt((DSD->PRP.Lx*DSD->PRP.Lx + DSD->PRP.Ly*DSD->PRP.Ly)/(2.0*DSD->PRP.Lx*DSD->PRP.Lx));
    DSD->PRP.ndn_Kappa_yy = 2.0*DSD->PRP.Kappa_yy / (R*T*DSD->PRP.Cm*(DSD->PRP.Lx*DSD->PRP.Lx + DSD->PRP.Ly*DSD->PRP.Ly)) *
                            sqrt((DSD->PRP.Lx*DSD->PRP.Lx + DSD->PRP.Ly*DSD->PRP.Ly)/(2.0*DSD->PRP.Ly*DSD->PRP.Ly));

    DSD->PRP.ndn_Omega = DSD->PRP.Omega * NA / (R*T);
    DSD->PRP.ndn_dx = DSD->PRP.dx / DSD->PRP.Lx;
    DSD->PRP.ndn_dy = DSD->PRP.dy / DSD->PRP.Ly;
    DSD->PRP.ndn_C0 = DSD->PRP.C0 / DSD->PRP.Cm;
    DSD->PRP.ndn_rndCAmp = DSD->PRP.rndCAmp / DSD->PRP.Cm;
    DSD->PRP.ndn_S0 = DSD->PRP.S0 / DSD->PRP.Cm * DSD->PRP.tau;
    DSD->PRP.ndn_C_rate = convert_Crate_per_second_to_ndn(DSD, DSD->PRP.C_rate);
    DSD->PRP.ndn_S_TOT = DSD->PRP.C_rate * N_CV_x * N_CV_y * DSD->PRP.tau;
    DSD->PRP.ndn_pulseTime = DSD->PRP.pulseTime / DSD->PRP.tau;
    DSD->PRP.ndn_relaxTime = DSD->PRP.relaxTime / DSD->PRP.tau;
    DSD->PRP.ndn_C_DL = DSD->PRP.C_DL * R*T / (F_CONST*F_CONST * DSD->PRP.Cm * DSD->PRP.dp);
    DSD->PRP.ndn_B_strain = DSD->PRP.B_strain / (R*T*DSD->PRP.Cm);

}


void calculate_CV_master_indeces (data_spinodal_decomposition DSD){
    int i,j;  //counter
    int i_CV = 0;  //counter
    int i_Ncv_x = N_CV_x;           //number of control volumes in x-direction
    int i_Ncv_y = N_CV_y;           //number of control volumes in y-direction


    for (i=0; i<i_Ncv_x; i++){
        for (j=0; j<i_Ncv_y; j++){

            DSD->CV[i_CV].pos_CV_P = i_CV;
            DSD->CV[i_CV].pos_CV_E = i_CV + 1;
            DSD->CV[i_CV].pos_CV_W = i_CV - 1;
            DSD->CV[i_CV].pos_CV_N = i_CV - i_Ncv_y;
            DSD->CV[i_CV].pos_CV_S = i_CV + i_Ncv_y;

            //boundary control volumes
            if (i == 0){
                DSD->CV[i_CV].pos_CV_N = i_CV + (i_Ncv_x-1) * i_Ncv_y;
            }

            if (i == i_Ncv_x-1){
                DSD->CV[i_CV].pos_CV_S = i_CV - (i_Ncv_x-1) * i_Ncv_y;
            }

            if (j == 0){
                DSD->CV[i_CV].pos_CV_W = i_CV + (i_Ncv_y - 1);
            }

            if (j == i_Ncv_y-1){
                DSD->CV[i_CV].pos_CV_E = i_CV - (i_Ncv_y - 1);
            }

            i_CV++;
        }
    }

}

void calculate_CV_xy_coordinates (data_spinodal_decomposition DSD){
    int i,j;  //counter
    int i_CV = 0;  //counter
    int i_Ncv_x = N_CV_x;           //number of control volumes in x-direction
    int i_Ncv_y = N_CV_y;           //number of control volumes in y-direction


    for (i=0; i<i_Ncv_x; i++){
        for (j=0; j<i_Ncv_y; j++){

            //centroids of each control volume in dimensional units
            DSD->CV[i_CV].xCV = DSD->PRP.dx * (0.5 + i);
            DSD->CV[i_CV].yCV = DSD->PRP.dy * (0.5 + j);

            //centroids of each control volume in non-dimensional units
            DSD->CV[i_CV].ndn_xCV = DSD->PRP.ndn_dx * (0.5 + i);
            DSD->CV[i_CV].ndn_yCV = DSD->PRP.ndn_dy * (0.5 + j);

            i_CV++;
        }
    }

}

void update_local_state_vector (N_Vector yy, data_spinodal_decomposition DSD){
    int i,j;  //counter
    int tmp = 0;  //counter
    int i_Ncv = N_CV_x*N_CV_y;      //total number of control volumes
    int i_N_var_tot = N_var_tot;    //number of state variables in each control volume
    int i_N_var_LSC_act = N_var_LSC_act;//number of active state variables in lithium source control equation

    realtype *yval;
    yval = N_VGetArrayPointer(yy);

    //control volumes
    for (i=0; i<i_Ncv; i++){
        for (j=0; j<i_N_var_tot; j++){
            DSD->CV[i].sv_CV[j] = yval[tmp];
            tmp++;
        }
    }

    //lithium source control
    for (i=0; i<i_N_var_LSC_act; i++){
        DSD->LSC.sv_LSC[i] = yval[tmp];
        tmp++;
    }
}


void calculate_local_jacobian_matrix (realtype cj, data_spinodal_decomposition DSD){
    int i_Ncv = N_CV_x*N_CV_y;          //total number of control volumes
    int sv_C = DSD->pos_SV_C;           //integer position of the concentration in the local state vector
    int sv_mu = DSD->pos_SV_mu;         //integer position of the chemical potential in the local state vector
    int sv_dPhi = DSD->pos_LSC_dPhi;    //integer position of the voltage difference in the local state vector
    int der_C = DSD->pos_deriv_C;       //integer position of the derivative over concentration
    int der_mu = DSD->pos_deriv_mu;     //integer position of the derivative over chemical potential
    int der_dPhi = DSD->pos_deriv_dPhi; //integer position of the derivative over voltage difference


    double const Chixx_by_dx2 = DSD->PRP.ndn_Chi_xx / (DSD->PRP.ndn_dx*DSD->PRP.ndn_dx); //pre-calculated Chixx/(dx^2) for computational efficiency
    double const Chiyy_by_dy2 = DSD->PRP.ndn_Chi_yy / (DSD->PRP.ndn_dy*DSD->PRP.ndn_dy); //pre-calculated Chiyy/(dy^2) for computational efficiency
    double const kappaxx_by_dx2 = DSD->PRP.ndn_Kappa_xx / (DSD->PRP.ndn_dx*DSD->PRP.ndn_dx); //pre-calculated kappa_xx/(dx^2) for computational efficiency
    double const kappayy_by_dy2 = DSD->PRP.ndn_Kappa_yy / (DSD->PRP.ndn_dy*DSD->PRP.ndn_dy); //pre-calculated kappa_yy/(dy^2) for computational efficiency

    //integer position of the partial derivative in local jacobian matrix of rhs function over variables in the central and neighbouring control volumes and across dPhi [-]
    int pos_P = DSD->pos_jac_P;
    int pos_E = DSD->pos_jac_E;
    int pos_W = DSD->pos_jac_W;
    int pos_N = DSD->pos_jac_N;
    int pos_S = DSD->pos_jac_S;
    int pos_dPhi = DSD->pos_jac_dPhi;

    double CP, CE, CW, CN, CS;      //variables for storing values of concentration in central and neighbouring control volumes
    double muP, muE, muW, muN, muS; //variables for storing values of chemical potential in central and neighbouring control volumes
    double dSP_dCP, dSP_dmuP, dSP_ddPhi; //variables for storing partial derivatives of the source term over concentration, chemical potential and potential difference in the central control volume

    int i_CV = 0; //control volume counter


    //loop across CVs
    for (i_CV=0; i_CV<i_Ncv; i_CV++){

        CP = DSD->CV[DSD->CV[i_CV].pos_CV_P].sv_CV[sv_C];
        CE = DSD->CV[DSD->CV[i_CV].pos_CV_E].sv_CV[sv_C];
        CW = DSD->CV[DSD->CV[i_CV].pos_CV_W].sv_CV[sv_C];
        CN = DSD->CV[DSD->CV[i_CV].pos_CV_N].sv_CV[sv_C];
        CS = DSD->CV[DSD->CV[i_CV].pos_CV_S].sv_CV[sv_C];

        muP = DSD->CV[DSD->CV[i_CV].pos_CV_P].sv_CV[sv_mu];
        muE = DSD->CV[DSD->CV[i_CV].pos_CV_E].sv_CV[sv_mu];
        muW = DSD->CV[DSD->CV[i_CV].pos_CV_W].sv_CV[sv_mu];
        muN = DSD->CV[DSD->CV[i_CV].pos_CV_N].sv_CV[sv_mu];
        muS = DSD->CV[DSD->CV[i_CV].pos_CV_S].sv_CV[sv_mu];

        dSP_dCP = DSD->CV[DSD->CV[i_CV].pos_CV_P].dS_ndn_dSVar[der_C];
        dSP_dmuP = DSD->CV[DSD->CV[i_CV].pos_CV_P].dS_ndn_dSVar[der_mu];
        dSP_ddPhi = DSD->CV[DSD->CV[i_CV].pos_CV_P].dS_ndn_dSVar[der_dPhi];


        /* --- CENTRAL position --- */
        ///dFC/dCP
        DSD->CV[i_CV].jac_CV[sv_C][sv_C][pos_P] =
                CP*(2.0-3.0*CP)*(Chixx_by_dx2*(muE-2.0*muP+muW) + Chiyy_by_dy2*(muN-2.0*muP+muS)) + dSP_dCP - cj;

        ///dFC/dmuP
        DSD->CV[i_CV].jac_CV[sv_C][sv_mu][pos_P] =
                Chixx_by_dx2*(CE*CE*(CE-1.0) + 2.0*CP*CP*(CP-1.0) + CW*CW*(CW-1.0)) +
                Chiyy_by_dy2*(CN*CN*(CN-1.0) + 2.0*CP*CP*(CP-1.0) + CS*CS*(CS-1.0)) + dSP_dmuP;

        ///dFC/ddPhi
        DSD->CV[i_CV].jac_CV[sv_C][sv_C][pos_dPhi] = dSP_ddPhi;

        ///dFmu/dCP
        DSD->CV[i_CV].jac_CV[sv_mu][sv_C][pos_P] =
                            1.0/(CP* (CP - 1.0)) +
                            2.0*DSD->PRP.ndn_Omega -
                            2.0*kappaxx_by_dx2 -
                            2.0*kappayy_by_dy2 -
                            DSD->PRP.ndn_B_strain;


        ///dFmu/dmuP
        DSD->CV[i_CV].jac_CV[sv_mu][sv_mu][pos_P] = 1.0;

        /* --- EAST position --- */
        ///dFC/dCE
        DSD->CV[i_CV].jac_CV[sv_C][sv_C][pos_E] = CE*(2.0-3.0*CE)*Chixx_by_dx2*(muE-muP);
        ///dFC/dmuE
        DSD->CV[i_CV].jac_CV[sv_C][sv_mu][pos_E] = Chixx_by_dx2*(CE*CE*(1.0-CE) + CP*CP*(1.0-CP));
        ///dFmu/dCE
        DSD->CV[i_CV].jac_CV[sv_mu][sv_C][pos_E] = kappaxx_by_dx2;
        ///dFmu/dmuE
        DSD->CV[i_CV].jac_CV[sv_mu][sv_mu][pos_E] = 0.0;

        /* --- WEST position --- */
        ///dFC/dCW
        DSD->CV[i_CV].jac_CV[sv_C][sv_C][pos_W] = CW*(2.0-3.0*CW)*Chixx_by_dx2*(muW-muP);
        ///dFC/dmuW
        DSD->CV[i_CV].jac_CV[sv_C][sv_mu][pos_W] = Chixx_by_dx2*(CW*CW*(1.0-CW) + CP*CP*(1.0-CP));
        ///dFmu/dCW
        DSD->CV[i_CV].jac_CV[sv_mu][sv_C][pos_W] = kappaxx_by_dx2;
        ///dFmu/dmuW
        DSD->CV[i_CV].jac_CV[sv_mu][sv_mu][pos_W] = 0.0;

        /* --- NORTH position --- */
        ///dFC/dCN
        DSD->CV[i_CV].jac_CV[sv_C][sv_C][pos_N] = CN*(2.0-3.0*CN)*Chiyy_by_dy2*(muN-muP);
        ///dFC/dmuN
        DSD->CV[i_CV].jac_CV[sv_C][sv_mu][pos_N] = Chiyy_by_dy2*(CN*CN*(1.0-CN) + CP*CP*(1.0-CP));
        ///dFmu/dCN
        DSD->CV[i_CV].jac_CV[sv_mu][sv_C][pos_N] = kappayy_by_dy2;
        ///dFmu/dmuN
        DSD->CV[i_CV].jac_CV[sv_mu][sv_mu][pos_N] = 0.0;

        /* --- SOUTH position --- */
        ///dFC/dCS
        DSD->CV[i_CV].jac_CV[sv_C][sv_C][pos_S] = CS*(2.0-3.0*CS)*Chiyy_by_dy2*(muS-muP);
        ///dFC/dmuS
        DSD->CV[i_CV].jac_CV[sv_C][sv_mu][pos_S] = Chiyy_by_dy2*(CS*CS*(1.0-CS) + CP*CP*(1.0-CP));
        ///dFmu/dCS
        DSD->CV[i_CV].jac_CV[sv_mu][sv_C][pos_S] = kappayy_by_dy2;
        ///dFmu/dmuS
        DSD->CV[i_CV].jac_CV[sv_mu][sv_mu][pos_S] = 0.0;

    }

    //lithium source control equation
    /* --- CENTRAL position --- */
    ///dF_LSC/ddPhi
    DSD->LSC.jac_LSC[sv_dPhi][sv_dPhi][sv_dPhi] = DSD->LSC.sum_dS_ndn_dSVar[DSD->pos_deriv_dPhi] - DSD->DLM.dS_DL_ndn_dSVar[DSD->pos_deriv_dPhi];

}

void initialize_master_IDA_vectors (N_Vector yy, N_Vector yp, N_Vector avtol, N_Vector id, data_spinodal_decomposition DSD){

    realtype *yval, *ypval, *atval, *idval;

    yval  = N_VGetArrayPointer(yy);
    ypval = N_VGetArrayPointer(yp);
    atval = N_VGetArrayPointer(avtol);
    idval = N_VGetArrayPointer(id);

    int i,j; //counter
    int tmp = 0;  //counter
    int i_Ncv = N_CV_x*N_CV_y;      //total number of control volumes
    int i_N_var_tot = N_var_tot;    //number of state variables in each control volume
    int i_N_var_LSC_act = N_var_LSC_act; //number of state variables in the lithium source control equation

    //control volumes
    for (i=0; i<i_Ncv; i++){
        for (j=0; j<i_N_var_tot; j++){
            yval[tmp] = DSD->CV[i].sv_CV[j];
            ypval[tmp] = 0.0; //initial value of d[state variable]/dt [./s]
            atval[tmp] = DSD->CV[i].atval_CV[j]; //absolute tolerances for state variables [.]
            idval[tmp] = DSD->CV[i].idval_CV[j]; //specification of algebraic/differential components in the y vector
            tmp++;
        }
    }

    //lithium source control equation
    for (i=0; i<i_N_var_LSC_act; i++){
        yval[tmp] = DSD->LSC.sv_LSC[i];
        ypval[tmp] = 0.0; //initial value of d[state variable]/dt [./s]
        atval[tmp] = DSD->LSC.atval_LSC[i]; //absolute tolerances for state variables [.]
        tmp++;
    }

}


void update_relative_absolute_tolerances (N_Vector avtol, data_spinodal_decomposition DSD){

    realtype *atval;

    atval = N_VGetArrayPointer(avtol);

    int i,j; //counter
    int tmp = 0;  //counter
    int i_Ncv = N_CV_x*N_CV_y;      //total number of control volumes
    int i_N_var_tot = N_var_tot;    //number of state variables in each control volume
    int i_N_var_LSC_act = N_var_LSC_act; //number of state variables in the lithium source control equation

    //update absolute tolerances for state vector variables
    for (i=0; i<i_Ncv; i++){
        DSD->CV[i].atval_CV[DSD->pos_SV_C] = ATOL_C; //absolute tolerance for non-dimensional concentration [-]
        DSD->CV[i].atval_CV[DSD->pos_SV_mu] = ATOL_mu; //absolute tolerance for non-dimensional chemical potential [-]
    }

    //update absolute tolerance for potential difference
    DSD->LSC.atval_LSC[DSD->pos_LSC_dPhi] = ATOL_dPhi; //absolute tolerance for non-dimensional voltage difference [-]


    //update IDA vector atval with updated absolute tolerance values
    //control volumes
    for (i=0; i<i_Ncv; i++){
        for (j=0; j<i_N_var_tot; j++){
            atval[tmp] = DSD->CV[i].atval_CV[j]; //absolute tolerances for state variables [.]
            tmp++;
        }
    }

    //lithium source control equation
    for (i=0; i<i_N_var_LSC_act; i++){
        atval[tmp] = DSD->LSC.atval_LSC[i]; //absolute tolerances for state variables [.]
        tmp++;
    }

    //update relative tolerance
    DSD->rtol = RTOL;

}

void write_jacobi_matrix (SUNMatrix JJ, data_spinodal_decomposition DSD){
    int i,j;
    int i_NEQ = NEQ;    //dimension of the matrix

    FILE *f = fopen(DSD->PFN.filename_jacobi, "w");
    if (f == NULL) {
        fprintf(stderr, "\nError opening file: %s, directory is not valid.\n\n", DSD->PFN.filename_jacobi);
        exit(1);
    }

    for (i=0; i<i_NEQ; i++){
        for (j=0; j<i_NEQ; j++){
            fprintf(f, "%g ", IJth(JJ,i,j));
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

void clear_output_files(data_spinodal_decomposition DSD){
    
    FILE *g = fopen(DSD->PFN.filename_species_conservation, "w");
    if (g == NULL) {
        fprintf(stderr, "\nError opening file: %s, directory is not valid.\n\n", DSD->PFN.filename_species_conservation);
        exit(1);
    }
    fclose(g);

    FILE *h = fopen(DSD->PFN.filename_stationary_level_mu, "w");
    if (h == NULL) {
        fprintf(stderr, "\nError opening file: %s, directory is not valid.\n\n", DSD->PFN.filename_stationary_level_mu);
        exit(1);
    }
    fclose(h);

    FILE *i = fopen(DSD->PFN.filename_output_Crate_profile, "w");
    if (i == NULL) {
        fprintf(stderr, "\nError opening file: %s, directory is not valid.\n\n", DSD->PFN.filename_output_Crate_profile);
        exit(1);
    }
    fclose(i);

    FILE *j = fopen(DSD->PFN.filename_potential_difference, "w");
    if (j == NULL) {
        fprintf(stderr, "\nError opening file: %s, directory is not valid.\n\n", DSD->PFN.filename_potential_difference);
        exit(1);
    }
    fclose(j);

    FILE *k = fopen(DSD->PFN.filename_animation_time_data, "w");
    if (k == NULL) {
        fprintf(stderr, "\nError opening file: %s, directory is not valid.\n\n", DSD->PFN.filename_animation_time_data);
        exit(1);
    }
    fclose(k);

    FILE *l = fopen(DSD->PFN.filename_average_chemical_potential, "w");
    if (l == NULL) {
        fprintf(stderr, "\nError opening file: %s, directory is not valid.\n\n", DSD->PFN.filename_average_chemical_potential);
        exit(1);
    }
    fclose(l);

    FILE *m = fopen(DSD->PFN.filename_clock_timestep, "w");
    if (m == NULL) {
        fprintf(stderr, "\nError opening file: %s, directory is not valid.\n\n", DSD->PFN.filename_clock_timestep);
        exit(1);
    }
    fclose(m);

}


void write_state_variable_matrix_form (data_spinodal_decomposition DSD, int pos_SV_write, int timeStepIndx){
    int i,j;  //counter
    int i_CV = 0;  //counter
    int i_Ncv_x = N_CV_x;               //number of control volumes in x-direction
    int i_Ncv_y = N_CV_y;               //number of control volumes in y-direction

    FILE *f = NULL;

    //select filename based on the function arguments
    if (timeStepIndx == 0){
        if (pos_SV_write == DSD->pos_SV_C){
            f = fopen(DSD->PFN.filename_matrix_concentration_start, "w");
        } else if (pos_SV_write == DSD->pos_SV_mu){
            f = fopen(DSD->PFN.filename_matrix_chem_potential_start, "w");
        }

    } else if (timeStepIndx == DSD->NOUT-1){
        if (pos_SV_write == DSD->pos_SV_C){
            f = fopen(DSD->PFN.filename_matrix_concentration_end, "w");
        } else if (pos_SV_write == DSD->pos_SV_mu){
            f = fopen(DSD->PFN.filename_matrix_chem_potential_end, "w");
        }
    }


    for (i=0; i<i_Ncv_x; i++){
        for (j=0; j<i_Ncv_y; j++){
            fprintf(f,"%g ", DSD->CV[i_CV].sv_CV[pos_SV_write]);

            i_CV++;
        }
        fprintf(f,"\n");
    }

    fclose(f);
}


void write_animation_results (data_spinodal_decomposition DSD){
    int i,j;  //counter
    int i_CV = 0;  //counter
    int i_Ncv_x = N_CV_x;               //number of control volumes in x-direction
    int i_Ncv_y = N_CV_y;               //number of control volumes in y-direction

    FILE *file_C = NULL;
    FILE *file_mu = NULL;
    FILE *file_support_data = NULL;
    FILE *file_time_data = NULL;

    file_C = fopen(DSD->PFN.filename_animation_concentration, "w");
    file_mu = fopen(DSD->PFN.filename_animation_chem_potential, "w");
    file_support_data = fopen(DSD->PFN.filename_animation_supporting_data, "w");
    file_time_data = fopen(DSD->PFN.filename_animation_time_data, "a");

    for (i=0; i<i_Ncv_x; i++){
        for (j=0; j<i_Ncv_y; j++){
            fprintf(file_C,"%.16f %.16f %.16f \n", DSD->CV[i_CV].ndn_xCV, DSD->CV[i_CV].ndn_yCV, DSD->CV[i_CV].sv_CV[DSD->pos_SV_C]);
            fprintf(file_mu,"%.16f %.16f %.16f \n", DSD->CV[i_CV].ndn_xCV, DSD->CV[i_CV].ndn_yCV, DSD->CV[i_CV].sv_CV[DSD->pos_SV_mu]);

            //update max and min value of chemical potential in entire domain
            if (DSD->mu_max < DSD->CV[i_CV].sv_CV[DSD->pos_SV_mu]){
                DSD->mu_max = DSD->CV[i_CV].sv_CV[DSD->pos_SV_mu];
            }
            if (DSD->mu_min > DSD->CV[i_CV].sv_CV[DSD->pos_SV_mu]){
                DSD->mu_min = DSD->CV[i_CV].sv_CV[DSD->pos_SV_mu];
            }

            i_CV++;
        }
        fprintf(file_C,"\n");
        fprintf(file_mu,"\n");
    }

    fprintf(file_time_data,"%i %.3f %.3f %.3f %.3f\n", DSD->animationIndx,
            convert_ndn_to_real_time(DSD, DSD->ndn_simTime),
            convert_Crate_unit_per_second_to_per_hour (DSD->PRP.C_rate),
            DSD->ndn_avg_C, DSD->ndn_avg_mu);

    fprintf(file_support_data,"%i\n%g\n%g\n", DSD->animationIndx,
            DSD->mu_min, DSD->mu_max);


    fclose(file_C);
    fclose(file_mu);
    fclose(file_time_data);
    fclose(file_support_data);

}

void update_time_step (data_spinodal_decomposition DSD, long int numStep_old, long int numStep_new){
    double dtFactor = 2.0 - 0.942459*pow(numStep_new-numStep_old, 0.161239); //empiric correlation
    if (DSD->PRP.ndn_dt < dt_MAX) DSD->PRP.ndn_dt *= dtFactor; //adjust time step with the dtFactor with limit at dt_MAX
}

void write_params_and_sim_stats(void *mem, data_spinodal_decomposition DSD){
    int retval;
    long int nst, nni, nje, nre, nreLS, netf, ncfn, nge;

    retval = IDAGetNumSteps(mem, &nst);
    check_retval(&retval, "IDAGetNumSteps", 1);
    retval = IDAGetNumResEvals(mem, &nre);
    check_retval(&retval, "IDAGetNumResEvals", 1);
    retval = IDAGetNumJacEvals(mem, &nje);
    check_retval(&retval, "IDAGetNumJacEvals", 1);
    retval = IDAGetNumNonlinSolvIters(mem, &nni);
    check_retval(&retval, "IDAGetNumNonlinSolvIters", 1);
    retval = IDAGetNumErrTestFails(mem, &netf);
    check_retval(&retval, "IDAGetNumErrTestFails", 1);
    retval = IDAGetNumNonlinSolvConvFails(mem, &ncfn);
    check_retval(&retval, "IDAGetNumNonlinSolvConvFails", 1);
    retval = IDAGetNumLinResEvals(mem, &nreLS);
    check_retval(&retval, "IDAGetNumLinResEvals", 1);
    retval = IDAGetNumGEvals(mem, &nge);
    check_retval(&retval, "IDAGetNumGEvals", 1);

    FILE *f = fopen(DSD->PFN.filename_parameters_and_statistic, "w");


    fprintf(f, "-------------------------Simulation parameters-----------------------------\n");
    fprintf(f, "Grid size:                                                  %i x %i\n", N_CV_x, N_CV_y);
    fprintf(f, "Number of time steps:                                       %i\n", DSD->NOUT);
    fprintf(f, "Length of the domain in the x-direction [m]:                %g\n", DSD->PRP.Lx);
    fprintf(f, "Width of the control volume in the x-direction [m]:         %g\n", DSD->PRP.dx);
    fprintf(f, "Length of the domain in the y-direction [m]:                %g\n", DSD->PRP.Ly);
    fprintf(f, "Width of the control volume in the y-direction [m]:         %g\n", DSD->PRP.dy);
    fprintf(f, "Time step [s]:                                              %g\n", DSD->PRP.dt);
    fprintf(f, "Maximal allowed time step [s]:                              %g\n", dt_MAX);
    fprintf(f, "Regular solution parameter [-]:                             %g\n", DSD->PRP.ndn_Omega);
    fprintf(f, "Initial concentration [mol/m^3]:                            %g\n", DSD->PRP.C0);
    fprintf(f, "Maximal concentration [mol/m^3]:                            %g\n", DSD->PRP.Cm);
    fprintf(f, "Diffusion constant in the x-direction [m^2/s]:              %g\n", DSD->PRP.D_xx);
    fprintf(f, "Diffusion constant in the x-direction [m^2/s]:              %g\n", DSD->PRP.D_yy);
    fprintf(f, "Strain in the x-direction [J/m]:                            %g\n", DSD->PRP.Kappa_xx);
    fprintf(f, "Strain in the y-direction [J/m]:                            %g\n", DSD->PRP.Kappa_yy);
    fprintf(f, "Random perturbation around initial concentration [mol/m^3]: %g\n", DSD->PRP.rndCAmp);
    fprintf(f, "Particle thickness in z-direction [m]:                      %g\n", DSD->PRP.dp);
    fprintf(f, "Anodic transfer coefficient [-]:                            %g\n", DSD->PRP.alpha_a);
    fprintf(f, "Cathodic transfer coefficient [-]:                          %g\n", DSD->PRP.alpha_c);
    fprintf(f, "Exchange current density in BV pre-factor [A/m^2]:          %g\n", DSD->PRP.i0);
    fprintf(f, "Double layer capacitance [F/m^2]:                           %g\n", DSD->PRP.C_DL);
    fprintf(f, "Strain [Pa]:                                                %g\n", DSD->PRP.B_strain);
    if (USE_source_term > 0){
        fprintf(f, "C-rate [1/h]:                                               %g\n", C_rate_);
        fprintf(f, "Duration of the pulse  [s]:                                 %g\n", DSD->PRP.pulseTime);
        fprintf(f, "Duration of the relaxation in-between pulses  [s]:          %g\n", DSD->PRP.relaxTime);
    }
    fprintf(f, "Relative tolerance [-]:                                     %g\n", RTOL);
    fprintf(f, "Absolute tolerance for non-dimensional concentration [-]:   %g\n", ATOL_C);
    fprintf(f, "Absolute tolerance for non-dimensional chem. potential [-]: %g\n", ATOL_mu);
    fprintf(f, "Absolute tolerance for non-dimensional potential diff. [-]: %g\n", ATOL_dPhi);


    fprintf(f,"\n-----------------------Simulation settings---------------------------------\n");
    fprintf(f, "USE_variable_timestep              = %i\n", USE_variable_timestep);
    fprintf(f, "USE_restart_simulation             = %i\n", USE_restart_simulation);
    fprintf(f, "USE_write_animation_files          = %i\n", USE_write_animation_files);
    fprintf(f, "USE_update_tols_after_n_dts        = %i\n", USE_update_tols_after_n_dts);
    fprintf(f, "USE_constant_initial_C_mu          = %i\n", USE_constant_initial_C_mu);
    fprintf(f, "USE_IVC_calculation                = %i\n", USE_IVC_calculation);
    fprintf(f, "Write state vector every           = %i time steps\n", writeStep_state_vector);
    fprintf(f, "Write species conservation every   = %i time steps\n", writeStep_species_conservation);
    fprintf(f, "Write stationary level mu          = %i time steps\n", writeStep_stationary_level_mu);
    fprintf(f, "Write C-rate profile               = %i time steps\n", writeStep_Crate_profile);
    fprintf(f, "Write potential difference         = %i time steps\n", writeStep_potential_difference);
    fprintf(f, "Write animation files              = %i time steps\n", writeStep_animation_files);
    fprintf(f, "Calc. value of stationary level mu = %i time steps\n", step_calculate_stationary_level_mu);
    fprintf(f, "Max value of stationary level mu   = %g \n", stationary_level_mu_MAX);


    fprintf(f,"\n----------------------Final Run Statistics---------------------------------\n");
    if(USE_solver_type == USE_DENSE)    fprintf(f, "Solver type:                       = DENSE, with user-supplied Jacobian\n");
    if(USE_solver_type == USE_SPGMR)    fprintf(f, "Solver type:                       = SPGMR\n");
    if(USE_solver_type == USE_SPFGMR)   fprintf(f, "Solver type:                       = SPFGMR\n");
    if(USE_solver_type == USE_SPBCGS)   fprintf(f, "Solver type:                       = SPBCGS\n");
    if(USE_solver_type == USE_SPTFQMR)  fprintf(f, "Solver type:                       = SPTFQMR\n");
    if(USE_solver_type == USE_PCG)      fprintf(f, "Solver type:                       = PCG\n");
    fprintf(f, "Execution time: solver             = %g s\n",DSD->clock_solver);
    fprintf(f, "Real time factor:                  = %g \n", DSD->clock_solver/convert_ndn_to_real_time(DSD, DSD->ndn_simTime));
    fprintf(f, "Number of steps                    = %ld\n", nst);
    fprintf(f, "Number of residual evaluations     = %ld\n", nre+nreLS);
    fprintf(f, "Number of Jacobian evaluations     = %ld\n", nje);
    fprintf(f, "Number of nonlinear iterations     = %ld\n", nni);
    fprintf(f, "Number of error test failures      = %ld\n", netf);
    fprintf(f, "Number of nonlinear conv. failures = %ld\n", ncfn);
    fprintf(f, "Number of root fn. evaluations     = %ld\n", nge);

    fclose(f);
}


void write_species_conservation (data_spinodal_decomposition DSD){

    FILE *f = fopen(DSD->PFN.filename_species_conservation, "a");
    fprintf(f, "%.16f %.16f\n", convert_ndn_to_real_time(DSD, DSD->ndn_simTime), DSD->ndn_avg_C);
    fclose(f);

}

void write_average_chemical_potential (data_spinodal_decomposition DSD){

    FILE *f = fopen(DSD->PFN.filename_average_chemical_potential, "a");
    fprintf(f, "%.16f %.16f %.16f\n", convert_ndn_to_real_time(DSD, DSD->ndn_simTime), DSD->ndn_avg_C, DSD->ndn_avg_mu);
    fclose(f);

}


void calculate_average_ndn_domain_concentration_chemPotential (data_spinodal_decomposition DSD){
    int i_Ncv = N_CV_x*N_CV_y;  //total number of control volumes
    int i_CV;                   //control volume counter
    double sumSpecies = 0.0;    //variable for summation of the species
    double sumMu = 0.0;         //variable for summation of the non-dimensional chemical potential

    //loop across CVs
    for (i_CV=0; i_CV<i_Ncv; i_CV++){
        sumSpecies += DSD->CV[i_CV].sv_CV[DSD->pos_SV_C];
        sumMu += DSD->CV[i_CV].sv_CV[DSD->pos_SV_mu];
    }

    DSD->ndn_avg_C = sumSpecies/i_Ncv;
    DSD->ndn_avg_mu = sumMu/i_Ncv;

}


void write_stationary_level_mu (data_spinodal_decomposition DSD){

    FILE *f = fopen(DSD->PFN.filename_stationary_level_mu, "a");
    fprintf(f, "%.16f %.16f %.16f\n", convert_ndn_to_real_time(DSD, DSD->ndn_simTime), DSD->ndn_avg_C,
            DSD->stationary_level_mu);
    fclose(f);

}

void write_Crate_profile (data_spinodal_decomposition DSD){

    FILE *f = fopen(DSD->PFN.filename_output_Crate_profile, "a");
    fprintf(f, "%.16f %.16f\n", convert_ndn_to_real_time(DSD, DSD->ndn_simTime),
            convert_Crate_unit_per_second_to_per_hour (DSD->PRP.C_rate));
    fclose(f);

}

void write_potential_difference (data_spinodal_decomposition DSD){

    FILE *f = fopen(DSD->PFN.filename_potential_difference, "a");
    fprintf(f, "%.16f %.16f %.16f\n", convert_ndn_to_real_time(DSD, DSD->ndn_simTime), DSD->ndn_avg_C,
            DSD->LSC.sv_LSC[DSD->pos_LSC_dPhi]);
    fclose(f);

}


void write_clock_timestep (data_spinodal_decomposition DSD){

    FILE *f = fopen(DSD->PFN.filename_clock_timestep, "a");
    fprintf(f, "%.16f %.16f\n", convert_ndn_to_real_time(DSD, DSD->ndn_simTime), DSD->clock_timestep);
    fclose(f);

}


double convert_ndn_to_real_time (data_spinodal_decomposition DSD, double ndn_time){
    //conversion from non*dimensional time to dimensional time
    return ndn_time * DSD->PRP.tau;
}

double convert_real_time_to_ndn (data_spinodal_decomposition DSD, double real_time){
    //conversion from dimensional time to the non-dimensional time
    return real_time / DSD->PRP.tau;
}

double convert_real_Li_source_to_ndn (data_spinodal_decomposition DSD, double real_S){
    //conversion from dimensional source of Li to the non-dimensional source of Li
    return real_S / DSD->PRP.Cm * DSD->PRP.tau;
}

double convert_Crate_per_second_to_ndn_Li_source (data_spinodal_decomposition DSD, double Crate_per_second){
    //conversion from C-rate [1/s] to the non-dimensional Li source into the entire 2D domain
    return Crate_per_second * N_CV_x * N_CV_y * DSD->PRP.tau;
}

double convert_Crate_unit_per_hour_to_per_second (double Crate_per_hour){
    return Crate_per_hour * 0.00027777777777778;
}

double convert_Crate_unit_per_second_to_per_hour (double Crate_per_second){
    return Crate_per_second * 3600.0;
}

double convert_Crate_per_second_to_ndn (data_spinodal_decomposition DSD, double Crate_per_second){
    return convert_Crate_unit_per_second_to_per_hour (Crate_per_second) / convert_real_time_to_ndn(DSD, 3600.0);
}

double convert_ndn_to_real_dPhi (data_spinodal_decomposition DSD, double ndn_dPhi){
    //conversion from non-dimensional potential difference to the dimensional potential difference in units of [V]
    return ndn_dPhi*R*T/F_CONST;
}

void set_paths_output_files (data_spinodal_decomposition DSD){
    char *filenameExtension = ".txt";       //text file extension

    char *file_species_conservation = "species_conservation";
    char *file_jacobi = "jacobi";
    char *file_matrix_concentration_start = "concentration_start";
    char *file_matrix_concentration_end = "concentration_end";
    char *file_matrix_chem_potential_start = "chem_pot_start";
    char *file_matrix_chem_potential_end = "chem_pot_end";
    char *file_parameters_and_statistic = "params_and_stats";
    char *file_stationary_level_mu = "stationary_level_mu";
    char *file_restart_file = "restart_file";
    char *file_output_Crate_profile = "Crate_profile";
    char *file_potential_difference = "potential_difference";
    char *file_animation_supporting_data = "anim_supporting_data";
    char *file_animation_time_data = "anim_time_data";
    char *file_average_chemical_potential = "average_mu";
    char *file_clock_timestep = "clock_timestep";


    //concatenate strings to path+filename
    snprintf(DSD->PFN.filename_species_conservation, Len_filename*sizeof(char), "%s%s%s", DSD->resultPath, file_species_conservation, filenameExtension);
    snprintf(DSD->PFN.filename_jacobi, Len_filename*sizeof(char), "%s%s%s", DSD->resultPath, file_jacobi, filenameExtension);
    snprintf(DSD->PFN.filename_matrix_chem_potential_start, Len_filename*sizeof(char), "%s%s%s", DSD->resultPath, file_matrix_chem_potential_start, filenameExtension);
    snprintf(DSD->PFN.filename_matrix_chem_potential_end, Len_filename*sizeof(char), "%s%s%s", DSD->resultPath, file_matrix_chem_potential_end, filenameExtension);
    snprintf(DSD->PFN.filename_matrix_concentration_start, Len_filename*sizeof(char), "%s%s%s", DSD->resultPath, file_matrix_concentration_start, filenameExtension);
    snprintf(DSD->PFN.filename_matrix_concentration_end, Len_filename*sizeof(char), "%s%s%s", DSD->resultPath, file_matrix_concentration_end, filenameExtension);
    snprintf(DSD->PFN.filename_parameters_and_statistic, Len_filename*sizeof(char), "%s%s%s", DSD->resultPath, file_parameters_and_statistic, filenameExtension);
    snprintf(DSD->PFN.filename_stationary_level_mu, Len_filename*sizeof(char), "%s%s%s", DSD->resultPath, file_stationary_level_mu, filenameExtension);
    snprintf(DSD->PFN.filename_restart_file, Len_filename*sizeof(char), "%s%s%s", DSD->resultPath, file_restart_file, filenameExtension);
    snprintf(DSD->PFN.filename_output_Crate_profile, Len_filename*sizeof(char), "%s%s%s", DSD->resultPath, file_output_Crate_profile, filenameExtension);
    snprintf(DSD->PFN.filename_potential_difference, Len_filename*sizeof(char), "%s%s%s", DSD->resultPath, file_potential_difference, filenameExtension);
    snprintf(DSD->PFN.filename_animation_supporting_data, Len_filename*sizeof(char), "%s%s%s", DSD->animationResultPath, file_animation_supporting_data, filenameExtension);
    snprintf(DSD->PFN.filename_animation_time_data, Len_filename*sizeof(char), "%s%s%s", DSD->animationResultPath, file_animation_time_data, filenameExtension);
    snprintf(DSD->PFN.filename_average_chemical_potential, Len_filename*sizeof(char), "%s%s%s", DSD->resultPath, file_average_chemical_potential, filenameExtension);
    snprintf(DSD->PFN.filename_clock_timestep, Len_filename*sizeof(char), "%s%s%s", DSD->resultPath, file_clock_timestep, filenameExtension);

}


void set_paths_output_animation_files (data_spinodal_decomposition DSD){
    char *filenameExtension = ".txt";       //text file extension

    char *file_animation_result_concentration = "concentration";
    char *file_animation_result_chemical_potential = "mu";

    //concatenate strings to path+filename
    snprintf(DSD->PFN.filename_animation_concentration, Len_filename*sizeof(char), "%s%s_%i%s", DSD->animationResultPath, file_animation_result_concentration, DSD->animationIndx, filenameExtension);
    snprintf(DSD->PFN.filename_animation_chem_potential, Len_filename*sizeof(char), "%s%s_%i%s", DSD->animationResultPath, file_animation_result_chemical_potential, DSD->animationIndx, filenameExtension);

}


int randomRangeInt (int minVal, int maxVal){
    int tmp = 0;    //temporary value

    if (minVal>maxVal){ //check if min value argument exceeds max value argument
        tmp = minVal;
        minVal = maxVal; //replace values
        maxVal = tmp;
    }
    return minVal + rand()%(maxVal - minVal + 1); //+1 is added to include entire interval [minR maxR]
}

double randomRangeDouble (double minVal, double maxVal){
   return minVal + (double)rand() / ((double)RAND_MAX) * (maxVal - minVal);
}


void calculate_stationary_state_level_chemical_potential(data_spinodal_decomposition DSD)
{

    int i_Ncv = N_CV_x*N_CV_y;      //total number of control volumes
    double muP, muE, muW, muN, muS;  //variables for storing values of chemical potential in central and neighbouring control volumes

    int sv_mu = DSD->pos_SV_mu; //integer position of the chemical potential in the local state vector

    double const one_by_dx2 = 1.0 / (DSD->PRP.ndn_dx*DSD->PRP.ndn_dx); //pre-calculated 1.0/(dx^2) for computational efficiency
    double const one_by_dy2 = 1.0 / (DSD->PRP.ndn_dy*DSD->PRP.ndn_dy); //pre-calculated 1.0/(dy^2) for computational efficiency

    double sumNablaMu = 0.0;    //variable for summation of discretised expression for sum_CV[abs(\nabla^2 mu)]

    int i_CV = 0; //control volume counter

    //loop across CVs
    for (i_CV=0; i_CV<i_Ncv; i_CV++){

        muP = DSD->CV[DSD->CV[i_CV].pos_CV_P].sv_CV[sv_mu];
        muE = DSD->CV[DSD->CV[i_CV].pos_CV_E].sv_CV[sv_mu];
        muW = DSD->CV[DSD->CV[i_CV].pos_CV_W].sv_CV[sv_mu];
        muN = DSD->CV[DSD->CV[i_CV].pos_CV_N].sv_CV[sv_mu];
        muS = DSD->CV[DSD->CV[i_CV].pos_CV_S].sv_CV[sv_mu];

        //summation
        sumNablaMu += fabs(one_by_dx2 * (muE - 2.0*muP + muW) + one_by_dy2 * (muN - 2.0*muP + muS));

    }

    //finally sum is divided by the number of control volumes
    sumNablaMu /= i_Ncv;

    //store value to the structure member
    DSD->stationary_level_mu = sumNablaMu;

}


void write_restart_file (data_spinodal_decomposition DSD){
    int i_Ncv = N_CV_x*N_CV_y;  //total number of control volumes
    int i_CV;                   //control volume counter
    int j;                      //counter
    int i_N_var_tot = N_var_tot;//number of state variables in each control volume
    int i_N_var_LSC_act = N_var_LSC_act;//number of active state variables in lithium source control equation

    FILE *f = fopen(DSD->PFN.filename_restart_file, "w");

    //loop across CVs
    for (i_CV=0; i_CV<i_Ncv; i_CV++){
        for (j=0; j<i_N_var_tot; j++){
            fprintf(f, "%.20f\n", DSD->CV[i_CV].sv_CV[j]);
        }
    }

    //lithium source control equation
    for (j=0; j<i_N_var_LSC_act; j++){
        fprintf(f, "%.20f\n", DSD->LSC.sv_LSC[j]);
    }

    //store non-dimensional time information
    fprintf(f, "%.20f\n", DSD->ndn_simTime);

    //close the file
    fclose(f);
}

void read_restart_file_and_initialize_state_vector (data_spinodal_decomposition DSD){
    int lineIndx = 0;   //counter for lines in the file
    int i_Ncv = N_CV_x*N_CV_y;  //total number of control volumes
    int i_N_var_tot = N_var_tot;//number of state variables in each control volume
    int i_N_var_LSC_act = N_var_LSC_act;//number of active state variables in lithium source control equation
    int i_CV;                   //control volume counter
    int j;                      //counter

    double tmpSV_value; //temporary value for SV value
    char line[256]; //character array for storing individual lines while reading file line by line

    FILE *file = fopen(DSD->PFN.filename_restart_file, "r");

    //check if file exists
    if (file == NULL) {
        fprintf(stderr, "\nError opening file: %s, directory is not valid.\n\n", DSD->PFN.filename_restart_file);
        exit(1);
    }

    //determine file length to check if it matches (number of control volumes) x (number of state variables) +one line containing ndn time
    while (fgets(line, sizeof(line), file) != NULL) { //read entire file and determine number of lines

		if(fscanf (file, "%lf", &tmpSV_value) > 0);
		lineIndx++;
    }
    fclose(file); //close file

    if (lineIndx != (i_Ncv*i_N_var_tot + i_N_var_LSC_act + 1)){
        fprintf(stderr, "Length of the restart file (=%i) does not match N_CV x N_var_tot + N_var_LSC_act + 1 (%i x %i + %i + 1 = %i)\n",
                lineIndx, i_Ncv, i_N_var_tot, N_var_LSC_act, i_Ncv*i_N_var_tot+i_N_var_LSC_act+1);
        exit(1);
    }

    //open file again, read it and initialize state vector
    FILE *file1 = fopen(DSD->PFN.filename_restart_file, "r");

    //control volumes
    for (i_CV=0; i_CV<i_Ncv; i_CV++){
        for (j=0; j<i_N_var_tot; j++){
            if(fscanf (file1, "%lf", &DSD->CV[i_CV].sv_CV[j]) > 0);
        }
    }

    //lithium source control equation
    for (j=0; j<i_N_var_LSC_act; j++){
        if(fscanf (file1, "%lf", &DSD->LSC.sv_LSC[DSD->pos_LSC_dPhi]) > 0);
    }


    //read last line where the non-dimensional time information is stored
    if(fscanf (file1, "%lf", &DSD->ndn_simTime) > 0);

    fclose(file1); //close file

    //print message to the console that simulation was restarted
    printf("\n=============================== RESTART ===============================\n");
}


void read_restart_animation_index (data_spinodal_decomposition DSD){

    FILE *file = fopen(DSD->PFN.filename_animation_supporting_data, "r");

    //check if file exists
    if (file == NULL) {
        fprintf(stderr, "\nError opening file: %s, directory is not valid.\n\n", DSD->PFN.filename_animation_supporting_data);
        exit(1);
    }

    //read the first value from the file
    if(fscanf (file, "%i", &DSD->animationIndx) > 0);

    //increase animation index by 1
    DSD->animationIndx++;

    fclose(file); //close file
}

void calculate_source_term (data_spinodal_decomposition DSD){
    int i_Ncv = N_CV_x*N_CV_y;      //total number of control volumes
    double CP, muP;  //variables for storing values of concentration and chemical potential in central control volume

    int sv_C = DSD->pos_SV_C;   //integer position of the concentration in the local state vector
    int sv_mu = DSD->pos_SV_mu; //integer position of the chemical potential in the local state vector

    int der_C = DSD->pos_deriv_C;       //integer position of the derivative over concentration
    int der_mu = DSD->pos_deriv_mu;     //integer position of the derivative over chemical potential
    int der_dPhi = DSD->pos_deriv_dPhi; //integer position of the derivative over voltage difference

    int pos_P;  //integer position of a control volume

    double const omega = DSD->PRP.ndn_Omega;

    double ndn_eta = 0.0;       //non-dimensional overpotential [-]
    double ndn_BVpf = 0.0;      //non-dimensional Butler-Volmer pre-factor
    double dndn_BVpf_dCP = 0.0; //temporary variable for storing partial derivative of BV pre-factor over concentration

    double sum_S_ndn = 0.0; //variable for summation of all source terms across control volumes
    double sum_dS_ndn_ddPhi = 0.0; //variable for summation of all partial derivatives of source terms over dPhi across control volumes

    int i_CV = 0; //control volume counter

    //loop across CVs
    for (i_CV=0; i_CV<i_Ncv; i_CV++){

        pos_P = DSD->CV[i_CV].pos_CV_P;

        CP  = DSD->CV[pos_P].sv_CV[sv_C];
        muP = DSD->CV[pos_P].sv_CV[sv_mu];

        ndn_eta = DSD->LSC.sv_LSC[DSD->pos_LSC_dPhi] + muP; //calculate overpotential

        //calculate cathodic and anodic exponential terms
        double const BV_exp_c = exp(-DSD->PRP.alpha_c * ndn_eta);
        double const BV_exp_a = exp((1.0-DSD->PRP.alpha_a) * ndn_eta);

        if (DSD->i_USE_source_term == 0){ //do not use source term;
            DSD->CV[pos_P].S_ndn = 0.0;
            DSD->CV[pos_P].dS_ndn_dSVar[der_C] = 0.0;
            DSD->CV[pos_P].dS_ndn_dSVar[der_mu] = 0.0;
            DSD->CV[pos_P].dS_ndn_dSVar[der_dPhi] = 0.0;

        } else if (DSD->i_USE_source_term == 2){ // S0 * BV_exp;

            //calculate BV pre-factor and its derivative over CP
            ndn_BVpf = DSD->PRP.ndn_S0;
            dndn_BVpf_dCP = 0.0;

        } else if (DSD->i_USE_source_term == 3){ // S0*sqrt(x(1-x)) * BV_exp;

            //calculate BV pre-factor and its derivative over CP
            ndn_BVpf = DSD->PRP.ndn_S0 * sqrt(CP*(1.0-CP));
            dndn_BVpf_dCP = DSD->PRP.ndn_S0 * (1.0-2.0*CP)/(2.0*sqrt(CP*(1.0-CP)));

        } else if (DSD->i_USE_source_term == 4){ // S0*sqrt(x(1-x)exp(omega*(1-2x))) * BV_exp;

            //calculate BV pre-factor and its derivative over CP
            ndn_BVpf = DSD->PRP.ndn_S0 * sqrt(CP*(1.0-CP)*exp(omega*(1.0-2.0*CP)));
            dndn_BVpf_dCP = DSD->PRP.ndn_S0 * (exp(omega*(1.0-2.0*CP))*(2.0*CP*CP*omega - 2.0*CP*(omega+1.0) + 1.0)) /
                                                (2.0*sqrt(CP*(1.0-CP)*exp(omega*(1.0-2.0*CP))));

        } else {
            fprintf(stderr, "\nSelected option for USE_source_term is not valid.\n\n");
            exit(1);
        }

        //store values to the DSD structure based on the selection of length of Taylor expansion series for exponential parts
        if (USE_linearised_source == 0){
            DSD->CV[pos_P].S_ndn = ndn_BVpf * (BV_exp_c - BV_exp_a);
            DSD->CV[pos_P].dS_ndn_dSVar[der_C] = dndn_BVpf_dCP * (BV_exp_c - BV_exp_a);
            DSD->CV[pos_P].dS_ndn_dSVar[der_mu] = ndn_BVpf * (-DSD->PRP.alpha_c * BV_exp_c - (1.0-DSD->PRP.alpha_a) * BV_exp_a);
            DSD->CV[pos_P].dS_ndn_dSVar[der_dPhi] = DSD->CV[pos_P].dS_ndn_dSVar[der_mu];

        } else if (USE_linearised_source == 1) { //linearised source term
            DSD->CV[pos_P].S_ndn = -ndn_BVpf * ndn_eta * (DSD->PRP.alpha_c + (1.0 - DSD->PRP.alpha_a));
            DSD->CV[pos_P].dS_ndn_dSVar[der_C] = -dndn_BVpf_dCP * ndn_eta * (DSD->PRP.alpha_c + (1.0 - DSD->PRP.alpha_a));
            DSD->CV[pos_P].dS_ndn_dSVar[der_mu] = -ndn_BVpf * (DSD->PRP.alpha_c + (1.0 - DSD->PRP.alpha_a));
            DSD->CV[pos_P].dS_ndn_dSVar[der_dPhi] = DSD->CV[pos_P].dS_ndn_dSVar[der_mu];

        } else if (USE_linearised_source == 2){ //quadratic source term
            DSD->CV[pos_P].S_ndn = ndn_BVpf * (-ndn_eta * (DSD->PRP.alpha_c + (1.0 - DSD->PRP.alpha_a)) +
               0.5*ndn_eta*ndn_eta*(DSD->PRP.alpha_c*DSD->PRP.alpha_c) - (1.0 - DSD->PRP.alpha_a)*(1.0 - DSD->PRP.alpha_a));

            DSD->CV[pos_P].dS_ndn_dSVar[der_C] = dndn_BVpf_dCP * (-ndn_eta * (DSD->PRP.alpha_c + (1.0 - DSD->PRP.alpha_a)) +
               0.5*ndn_eta*ndn_eta*(DSD->PRP.alpha_c*DSD->PRP.alpha_c - (1.0 - DSD->PRP.alpha_a)*(1.0 - DSD->PRP.alpha_a)));

            DSD->CV[pos_P].dS_ndn_dSVar[der_mu] = ndn_BVpf * (-(DSD->PRP.alpha_c + (1.0 - DSD->PRP.alpha_a)) +
                ndn_eta * (DSD->PRP.alpha_c*DSD->PRP.alpha_c - (1.0 - DSD->PRP.alpha_a)*(1.0 - DSD->PRP.alpha_a)));

            DSD->CV[pos_P].dS_ndn_dSVar[der_dPhi] = DSD->CV[pos_P].dS_ndn_dSVar[der_mu];

        } else if (USE_linearised_source == -1){ //piecewise function (BV + linear extrapolation for high eta)
            double abs_ndn_eta_switch = 10.0; ///xxx absolute value of non-dimensional overpotential where switch between BV and linear extrapolation is performed

            //low overpotentials -> use standard Butler-Volmer equation
            if (fabs(ndn_eta)<abs_ndn_eta_switch){
                DSD->CV[pos_P].S_ndn = ndn_BVpf * (BV_exp_c - BV_exp_a);
                DSD->CV[pos_P].dS_ndn_dSVar[der_C] = dndn_BVpf_dCP * (BV_exp_c - BV_exp_a);
                DSD->CV[pos_P].dS_ndn_dSVar[der_mu] = ndn_BVpf * (-DSD->PRP.alpha_c * BV_exp_c - (1.0-DSD->PRP.alpha_a) * BV_exp_a);
                DSD->CV[pos_P].dS_ndn_dSVar[der_dPhi] = DSD->CV[pos_P].dS_ndn_dSVar[der_mu];

            //high overpotential -> use linear extrapolation by calculation of the tangent in the abs_ndn_eta_switch point
            } else if (ndn_eta<-abs_ndn_eta_switch){
                double coeff_k = calculate_derivative_BV_exponent_over_eta(DSD, -abs_ndn_eta_switch);
                double coeff_n = calculate_BV_exponent(DSD, -abs_ndn_eta_switch) + coeff_k * abs_ndn_eta_switch;

                DSD->CV[pos_P].S_ndn = ndn_BVpf * (coeff_k * ndn_eta + coeff_n);
                DSD->CV[pos_P].dS_ndn_dSVar[der_C] = dndn_BVpf_dCP * (coeff_k * ndn_eta + coeff_n);
                DSD->CV[pos_P].dS_ndn_dSVar[der_mu] = ndn_BVpf * coeff_k;
                DSD->CV[pos_P].dS_ndn_dSVar[der_dPhi] = DSD->CV[pos_P].dS_ndn_dSVar[der_mu];


            } else if (ndn_eta>abs_ndn_eta_switch){
                double coeff_k = calculate_derivative_BV_exponent_over_eta(DSD, abs_ndn_eta_switch);
                double coeff_n = calculate_BV_exponent(DSD, abs_ndn_eta_switch) - coeff_k * abs_ndn_eta_switch;

                DSD->CV[pos_P].S_ndn = ndn_BVpf * (coeff_k * ndn_eta + coeff_n);
                DSD->CV[pos_P].dS_ndn_dSVar[der_C] = dndn_BVpf_dCP * (coeff_k * ndn_eta + coeff_n);
                DSD->CV[pos_P].dS_ndn_dSVar[der_mu] = ndn_BVpf * coeff_k;
                DSD->CV[pos_P].dS_ndn_dSVar[der_dPhi] = DSD->CV[pos_P].dS_ndn_dSVar[der_mu];
            }

        } else {
            fprintf(stderr, "\nSelected option for USE_linearised_source is not valid.\n\n");
            exit(1);
        }

        //summation of all source terms and its derivatives
        sum_S_ndn += DSD->CV[pos_P].S_ndn;
        sum_dS_ndn_ddPhi += DSD->CV[pos_P].dS_ndn_dSVar[der_dPhi];
    }

    //store the results of summation
    DSD->LSC.sum_S_ndn_CV = sum_S_ndn;
    DSD->LSC.sum_dS_ndn_dSVar[DSD->pos_deriv_dPhi] = sum_dS_ndn_ddPhi;
}

double calculate_derivative_BV_exponent_over_eta (data_spinodal_decomposition DSD, double ndn_eta_value){
    //calculate partial derivatives of cathodic and anodic exponential terms over ndn_eta
    double dBV_exp_c_dEta = -DSD->PRP.alpha_c * exp(-DSD->PRP.alpha_c * ndn_eta_value);
    double dBV_exp_a_dEta = (1.0-DSD->PRP.alpha_a) * exp((1.0-DSD->PRP.alpha_a) * ndn_eta_value);

    return dBV_exp_c_dEta - dBV_exp_a_dEta;
}

double calculate_BV_exponent (data_spinodal_decomposition DSD, double ndn_eta_value){
    //calculate cathodic and anodic exponential terms
    double BV_exp_c = exp(-DSD->PRP.alpha_c * ndn_eta_value);
    double BV_exp_a = exp((1.0-DSD->PRP.alpha_a) * ndn_eta_value);

    return BV_exp_c - BV_exp_a;
}

void generate_Li_source_pulses (data_spinodal_decomposition DSD){

    double wave = 0.0;              //value of sine wave at a given time with specified frequency
    double d_pi = PI_CONST;         //pi constant
    double frequency = frequency_;  //frequency of the waveform [Hz]
    double threshold = threshold_;  //threshold on the sinusoidal curve to switch between different modes of pulses (+, -, 0)

    if (USE_pulse_shape == 0){ //constant source of Li based on the initial selection (always in the same direction, either + or -)

        DSD->PRP.C_rate = convert_Crate_unit_per_hour_to_per_second(DSD->PRP.tmp_C_rate); //convert input unit [1/h] to [1/s]
        DSD->PRP.ndn_C_rate = convert_Crate_per_second_to_ndn(DSD, DSD->PRP.C_rate); //calculation of non-dimensional C-rate
        DSD->PRP.S_TOT = DSD->PRP.C_rate * DSD->PRP.Cm * N_CV_x * N_CV_y; //calculate total source term based on the C-rate
        DSD->PRP.ndn_S_TOT = convert_real_Li_source_to_ndn(DSD, DSD->PRP.S_TOT); //calculate non-dimensional total source term based on the C-rate

    } else if (USE_pulse_shape == 1){ //pulses of Li source (always in the same direction, either + or -) with relaxation periods in-between
        //       ___      ___      ___      ___
        // ______|+|______|+|______|+|______|+|______

        //calculate value of the sine wave at the current non-dimensional simulation time
        wave = sin(2.0*d_pi*frequency*DSD->ndn_simTime);

        if (wave > threshold){ //pulse stage
            DSD->PRP.C_rate = convert_Crate_unit_per_hour_to_per_second(DSD->PRP.tmp_C_rate); //convert input unit [1/h] to [1/s]
            DSD->PRP.ndn_C_rate = convert_Crate_per_second_to_ndn(DSD, DSD->PRP.C_rate); //calculation of non-dimensional C-rate
            DSD->PRP.S_TOT = DSD->PRP.C_rate * DSD->PRP.Cm * N_CV_x * N_CV_y; //calculate total source term based on the C-rate
            DSD->PRP.ndn_S_TOT = convert_real_Li_source_to_ndn(DSD, DSD->PRP.S_TOT); //calculate non-dimensional total source term based on the C-rate

        } else {//relaxation stage
            DSD->PRP.C_rate = 0.0;
            DSD->PRP.ndn_C_rate = 0.0;
            DSD->PRP.S_TOT = 0.0;
            DSD->PRP.ndn_S_TOT = 0.0;

        }

    } else if (USE_pulse_shape == 2){ //pulses of Li source in both directions with relaxation periods in-between
        //       ___               ___
        // ______|+|_______________|+|_______________
        //                |_|               |_|
        //

        //calculate value of the sine wave at the current non-dimensional simulation time
        wave = sin(2.0*d_pi*frequency*DSD->ndn_simTime);

        DSD->PRP.C_rate = convert_Crate_unit_per_hour_to_per_second(DSD->PRP.tmp_C_rate); //convert input unit [1/h] to [1/s]
        DSD->PRP.ndn_C_rate = convert_Crate_per_second_to_ndn(DSD, DSD->PRP.C_rate); //calculation of non-dimensional C-rate
        DSD->PRP.S_TOT = DSD->PRP.C_rate * DSD->PRP.Cm * N_CV_x * N_CV_y; //calculate total source term based on the C-rate
        DSD->PRP.ndn_S_TOT = convert_real_Li_source_to_ndn(DSD, DSD->PRP.S_TOT); //calculate non-dimensional total source term based on the C-rate

        if (wave > threshold){//pulse stage +
            DSD->PRP.C_rate *= -1.0;
            DSD->PRP.ndn_C_rate *= -1.0;
            DSD->PRP.S_TOT *= -1.0;
            DSD->PRP.ndn_S_TOT *= -1.0;
        } else if (wave < -threshold) { //pulse stage -
            DSD->PRP.C_rate *= -1.0;
            DSD->PRP.ndn_C_rate *= -1.0;
            DSD->PRP.S_TOT *= -1.0;
            DSD->PRP.ndn_S_TOT *= -1.0;
        } else {//relaxation stage
            DSD->PRP.C_rate = 0.0;
            DSD->PRP.ndn_C_rate = 0.0;
            DSD->PRP.S_TOT = 0.0;
            DSD->PRP.ndn_S_TOT = 0.0;
        }

    } else if (USE_pulse_shape == 3){ //read C-rate profile from external file

        //run event handler for input data in each time step and set C-rate or non-dimensional source term accordingly
        if (DSD->INDAT.dataLength > DSD->INDAT.dataIndx+1){ //check for length of the data

            event_handler_input_data(DSD); //call event handler function which sets proper C-rate and Li source based on the input file current profile

        } else { //enter relaxation period if the simulation time exceeds time defined in the input profile
            DSD->PRP.C_rate = 0.0;
            DSD->PRP.ndn_C_rate = 0.0;
            DSD->PRP.S_TOT = 0.0;
            DSD->PRP.ndn_S_TOT = 0.0;
        }

    } else if (USE_pulse_shape == 4){ //construct pulses by creating time events of switching the C-rate
        event_handler_pulse_generator(DSD); //call event handler function which generates upcoming events of starting the pulses or relaxation periods

    }

}


void read_input_current_profile (data_spinodal_decomposition DSD){
    int lineIndx = 0;   //counter for lines in the file
    int j;              //counter
    double tmp_value;   //temporary variable


    char line[256]; //character array for storing individual lines while reading file line by line

    FILE *file = fopen(DSD->PFN.filename_input_Crate_profile, "r");

    //check if file exists
    if (file == NULL) {
        fprintf(stderr, "\nError opening file: %s, directory is not valid or filename does not exist.\n\n", DSD->PFN.filename_input_Crate_profile);
        exit(1);
    }

    //determine file length
    while (fgets(line, sizeof(line), file) != NULL) { //read entire file and determine number of lines

		if(fscanf (file, "%lf", &tmp_value) > 0){
            lineIndx++;
		}


    }
    fclose(file); //close file

    //based on the length of input file, allocate memory to the time and C-rate arrays
    DSD->INDAT.time_data = calloc(lineIndx, sizeof(double));
    DSD->INDAT.Crate_data = calloc(lineIndx, sizeof(double));

    //store number of lines
    DSD->INDAT.dataLength = lineIndx;

    //open file again, read it and store data to the input data arrays
    FILE *file1 = fopen(DSD->PFN.filename_input_Crate_profile, "r");

    //store time and C-rate data to the array
    for (j=0; j<lineIndx; j++){
        if(fscanf (file1, "%lf %lf", &DSD->INDAT.time_data[j], &DSD->INDAT.Crate_data[j]) > 0);
    }
    fclose(file1); //close file

}


void event_handler_input_data (data_spinodal_decomposition DSD){
    double ndn_tEvent = convert_real_time_to_ndn(DSD, DSD->INDAT.time_data[DSD->INDAT.dataIndx+1]); //time of the next event

    if (DSD->INDAT.delaySwitch == 1){ //check if switch is activated to set the new C-rate with a single time step delay after the event was triggered
        //set non-dimensional source S based on the input data containing C-rate in units of [1/h]
        DSD->PRP.C_rate = convert_Crate_unit_per_hour_to_per_second(DSD->INDAT.Crate_data[DSD->INDAT.dataIndx]); //convert from [1/h] to [1/s]
        DSD->PRP.ndn_C_rate = convert_Crate_per_second_to_ndn(DSD, DSD->PRP.C_rate); //calculation of non-dimensional C-rate
        DSD->PRP.S_TOT = DSD->PRP.C_rate * DSD->PRP.Cm * N_CV_x * N_CV_y; //calculation of dimensional total source term [mol/m^3 s]
        DSD->PRP.ndn_S_TOT = convert_Crate_per_second_to_ndn_Li_source(DSD, DSD->PRP.C_rate); //calculation of non-dimensional total source term

        //set switch back to zero value
        DSD->INDAT.delaySwitch = 0;
    }

    if (DSD->ndn_simTime + DSD->PRP.ndn_dt > ndn_tEvent){
        DSD->PRP.ndn_dt = ndn_tEvent - DSD->ndn_simTime; //decrease time step in order to capture the event

        DSD->INDAT.delaySwitch = 1;

        DSD->INDAT.dataIndx++; //increase index for reading the data
    }
}


void event_handler_pulse_generator (data_spinodal_decomposition DSD){

    if (DSD->pulse_relax_ID == 1){
        DSD->PRP.C_rate = 0.0;
        DSD->PRP.ndn_C_rate = 0.0;
        DSD->PRP.S_TOT = 0.0;
        DSD->PRP.ndn_S_TOT = 0.0;

        if(USE_variable_timestep == 1){
            DSD->PRP.ndn_dt = VALUE_min(DSD->PRP.ndn_dt, 0.1 * DSD->PRP.ndn_relaxTime); //manually decrease time step in relaxation period to 1/10 of the relaxation duration

        } else { //reset time step to the starting value after it was decreased
            DSD->PRP.ndn_dt = convert_real_time_to_ndn(DSD, dt_);
        }

        DSD->pulse_relax_ID = 2; //set ID to pulse period
    }

    if (DSD->pulse_relax_ID == 3){
        DSD->PRP.C_rate = convert_Crate_unit_per_hour_to_per_second(DSD->PRP.tmp_C_rate); //convert from [1/h] to [1/s]
        DSD->PRP.ndn_C_rate = convert_Crate_per_second_to_ndn(DSD, DSD->PRP.C_rate); //calculation of non-dimensional C-rate
        DSD->PRP.S_TOT = DSD->PRP.C_rate * DSD->PRP.Cm * N_CV_x * N_CV_y; //calculation of dimensional total source term [mol/m^3 s]
        DSD->PRP.ndn_S_TOT = convert_Crate_per_second_to_ndn_Li_source(DSD,DSD->PRP.C_rate); //calculation of non-dimensional total source term

        if(USE_variable_timestep == 1){
            DSD->PRP.ndn_dt = VALUE_min(DSD->PRP.ndn_dt, 0.1 * DSD->PRP.ndn_pulseTime); //manually decrease time step in pulse to 1/10 of the pulse duration

        } else { //reset time step to the starting value after it was decreased
            DSD->PRP.ndn_dt = convert_real_time_to_ndn(DSD, dt_);
        }


        DSD->pulse_relax_ID = 0; //set ID to the relaxation period
    }

    if (DSD->ndn_simTime + DSD->PRP.ndn_dt > DSD->PRP.ndn_eventTime){
        DSD->PRP.ndn_dt = DSD->PRP.ndn_eventTime - DSD->ndn_simTime; //decrease time step in order to capture the event

        if (DSD->pulse_relax_ID == 0){ //current ID is set to relaxation period
            DSD->PRP.ndn_eventTime += DSD->PRP.ndn_relaxTime; //increase event time to the start of the next pulse

            DSD->pulse_relax_ID = 1; //set ID to 1 in order to set C-rate in the next step to relaxation
        }

        if (DSD->pulse_relax_ID == 2){ //current ID is set to pulse period
            DSD->PRP.ndn_eventTime += DSD->PRP.ndn_pulseTime; //increase event time to the start of the next relaxation

            DSD->pulse_relax_ID = 3; //set ID to 3 in order to set C-rate in the next step to pulse
        }

    }
}


void calc_source_term_double_layer(N_Vector yp, data_spinodal_decomposition DSD) {
    int i_Ncv = N_CV_x*N_CV_y;          //total number of control volumes
    int i_N_var_tot = N_var_tot;        //number of state variables in each control volume
    int dPhi_SV = i_Ncv * i_N_var_tot;  //variable for storing position of RHS_fun(voltage difference dPhi) in master state vector

	realtype *ypval;
    ypval = N_VGetArrayPointer(yp);

    DSD->DLM.S_DL_ndn = DSD->PRP.ndn_C_DL * ypval[dPhi_SV];

}


void calc_jac_double_layer(realtype cj, data_spinodal_decomposition DSD) {

    DSD->DLM.dS_DL_ndn_dSVar[DSD->pos_deriv_dPhi] = DSD->PRP.ndn_C_DL * cj;

}


void calculate_diagonal_Jac_elements (realtype cj, data_spinodal_decomposition DSD){

    //calculate jacobi contributions to double layer source term
    if (USE_double_layer_model == 1){
        calc_jac_double_layer(cj, DSD);
    }

    //calculate partial derivatives to local jacobi matrix
    calculate_local_jacobian_matrix(cj, DSD);

    int i_Ncv = N_CV_x*N_CV_y;      //total number of control volumes
    int i_N_var_tot = N_var_tot;    //number of state variables in each control volume

    int j,k; //counter

    //integer position of the partial derivative in local jacobian matrix of rhs function over variables in the central/east/west/north/south control volume [-]
    int pos_P = DSD->pos_jac_P;

    int sv_dPhi = DSD->pos_LSC_dPhi;    //integer position of the voltage difference in the local state vector
    int dPhi_JAC = i_Ncv * i_N_var_tot; //variable for storing position of lithium source control equation in JAC matrix

    int C_P = 0;    //variable for storing position of first state vector variable of central CV in the master rhs vector
    int i_CV = 0;   //control volume counter

    //loop across CVs
    for (i_CV=0; i_CV<i_Ncv; i_CV++){
        C_P = DSD->CV[i_CV].pos_CV_P * i_N_var_tot + DSD->pos_SV_C;

        for (j=0; j<i_N_var_tot; j++){
            for (k=0; k<i_N_var_tot; k++){
                //store only diagonal elements from calculated local Jacobian
                if (j==k) DSD->JacDiag[C_P+j] = DSD->CV[i_CV].jac_CV[j][k][pos_P];
            }
        }
    }

    //lithium source control - equation itself
    if (USE_source_term > 0){ //avoid memory leaks
        ///dF_LSC/ddPhi
        DSD->JacDiag[dPhi_JAC] =  DSD->LSC.jac_LSC[sv_dPhi][sv_dPhi][sv_dPhi];
    }
}


int PsetupSD(realtype tt,
               N_Vector yy, N_Vector yp, N_Vector rr,
               realtype cj, void *user_data)
{
    //setup for diagonal preconditioner

    int i;
    int i_NEQ = NEQ;    //dimension of the problem

    data_spinodal_decomposition DSD;
    DSD = (data_spinodal_decomposition)user_data;

    realtype *ppv;
    ppv = N_VGetArrayPointer(DSD->pp);

    //calculate diagonal Jacobi elements
    calculate_diagonal_Jac_elements(cj, DSD);

    for (i=0; i<i_NEQ; i++){
        //calculate inverse values of diagonal Jacobi elements
        ppv[i] = 1.0 / DSD->JacDiag[i];
    }

    return(0);
}


int PsolveSD(realtype tt,
               N_Vector yy, N_Vector yp, N_Vector rr,
               N_Vector rvec, N_Vector zvec,
               realtype cj, realtype delta, void *user_data)
{
    // solve preconditioner linear system P z = r

    data_spinodal_decomposition DSD;
    DSD = (data_spinodal_decomposition)user_data;
    N_VProd(DSD->pp, rvec, zvec);

    return(0);
}


void update_simulation_parameters (data_spinodal_decomposition DSD, double dt, double Nout){
    DSD->PRP.dt = dt;
    DSD->NOUT = Nout;
}

