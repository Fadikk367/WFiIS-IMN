#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mgmres.h"

// definicja stałych
#define delta 0.1
#define itr_max 500
#define mr 500
#define tol_abs pow(10, -8)
#define tol_rel pow(10, -8)

struct Potentials {
  double V1;
  double V2;
  double V3;
  double V4;
};

// struktura przechowująca wszystkie niezbędne parametry
struct Params {
  struct Potentials potentials;
  int nx;
  int ny;
  double x_max;
  double y_max;
  double eps1;
  double eps2;
};

void initialize_params(struct Params* params) {
  params->potentials.V1 = 10.0;
  params->potentials.V2 = -10.0;
  params->potentials.V3 = 10.0;
  params->potentials.V4 = -10.0;

  params->nx = 4;
  params->ny = 4;

  params->x_max = 0.0;
  params->y_max = 0.0;

  params->eps1 = 1.0;
  params->eps2 = 1.0;
}

void set_potentials(struct Params* params, double V1, double V2, double V3, double V4) {
  params->potentials.V1 = V1;
  params->potentials.V2 = V2;
  params->potentials.V3 = V3;
  params->potentials.V4 = V4;
}

void set_node_numbers(struct Params* params, int nx, int ny) {
  params->nx = nx;
  params->ny = ny;
}

void set_max_ranges(struct Params* params, double x_max, double y_max) {
  params->x_max = x_max;
  params->y_max = y_max;
}

void set_epsilons(struct Params* params, double eps1, double eps2) {
  params->eps1 = eps1;
  params->eps2 = eps2;
}


int N(int nx, int ny){
    return (nx+1)*(ny+1);
}

int j(int nx, int l){
    return (l/(nx+1));
}

int i(int nx, int l){
    return l-j(nx,l)*(nx+1);
}

double get_epsilon(int nx, int l, int eps1, int eps2){
  return i(nx, l) <= nx/2 ? eps1 : eps2;
}

double density_1(struct Params params, int l, double sigma) {
  double x = i(params.nx, l)*delta;
  double y = j(params.nx, l)*delta;
	return exp(-pow(x - 0.25*params.x_max, 2)/pow(sigma, 2) - pow(y - 0.5*params.y_max, 2)/pow(sigma, 2));
}

double density_2(struct Params params, int l, double sigma) {
  double x = i(params.nx, l)*delta;
  double y = j(params.nx, l)*delta;
	return -exp(-pow(x - 0.75*params.x_max, 2)/pow(sigma, 2) - pow(y - 0.5*params.y_max, 2)/pow(sigma, 2));
}

int fill_matrix(struct Params params, int *ja, int *ia, double *a, double *b, FILE* matrix_file, FILE* vector_file) {
  double sigma = params.x_max / 10.0;
	int k = -1; // numeruje niezerowe elemnty A
	int nz_num = 0; // ilosc niezerowych elementow

	for(int l = 0; l < N(params.nx, params.ny); ++l) {
		int brzeg = 0;  // wskaznik położenia: 0 - środek obszaru; 1 - brzeg
		double vb = 0;  // potencjal na brzegu

		if (i(params.nx, l) == 0) { // lewy brzeg
			brzeg = 1;
			vb = params.potentials.V1;
		}

		if (j(params.nx, l) == params.ny) { // górny brzeg
			brzeg = 1;
			vb = params.potentials.V2;
		}

		if (i(params.nx, l) == params.nx) { // prawy brzeg
			brzeg = 1;
			vb = params.potentials.V3;
		}

		if (j(params.nx, l) == 0) { // dolny brzeg
			brzeg = 1;
			vb = params.potentials.V4;
		}

    b[l] = (-1)*(density_1(params, l, sigma) + density_2(params, l, sigma));


		if (brzeg == 1) {
			b[l] = vb; // wymuszamy wartość potencjału na brzegu
    }

    // wypełniamy elementy macierzy A
		ia[l] = -1; // wskaźnik dla pierwszego el. w wierszu

		if (l - params.nx - 1 > 0 && brzeg == 0) {
			k++;
			if(ia[l] < 0) {
        ia[l] = k;
      }

			a[k] = get_epsilon(params.nx, l, params.eps1, params.eps2) / (delta*delta);
			ja[k] = l - params.nx - 1;
		}

    //poddiagonala
		if (l-1 > 0 && brzeg == 0) {
			k++;
			if(ia[l] < 0) {
        ia[l] = k;
      }

			a[k] = get_epsilon(params.nx, l, params.eps1, params.eps2) / (delta*delta);
			ja[k] = l - 1;
		}

    //diagonala
		k++;
		if (ia[l] < 0) {
      ia[l] = k;
    }

		if (brzeg == 0) {
			a[k] = -(2*get_epsilon(params.nx, l, params.eps1, params.eps2) + get_epsilon(params.nx, l+1, params.eps1, params.eps2) + get_epsilon(params.nx, l + params.nx+1, params.eps1, params.eps2)) / (delta*delta);
    } else {
			a[k] = 1;
    }

		ja[k] = l;

    //naddiagonala
		if (l < N(params.nx, params.ny) && brzeg == 0) {
			k++;
			a[k] = get_epsilon(params.nx, l+1, params.eps1, params.eps2) / (delta*delta);
			ja[k] = l + 1;
		}

    //prawa skrajna przekątna
		if (l < N(params.nx, params.ny) - params.nx - 1 && brzeg == 0) {
			k++;
			a[k] = get_epsilon(params.nx, l + params.nx + 1, params.eps1, params.eps2) / (delta*delta);
			ja[k] = l + params.nx + 1;
		}

    if (matrix_file) {
      fprintf(matrix_file,"%d %d %d %f \n", l, i(params.nx,l), j(params.nx,l), b[l]);
    }
  }

	nz_num = k+1;
	ia[N(params.nx, params.ny)] = nz_num;

  if (vector_file) {
    for(int z = 0; z < 5*N(params.nx, params.ny); z++) {
      fprintf(vector_file, "%d %0.f \n", z, a[z]);
    }
  }

  return nz_num;
}


void solve_poisson_equation(struct Params params, FILE* map_file, FILE* matrix_file, FILE* vector_file) {
  // alokacja pamięci dla wektorów
  double* a = (double*) malloc(sizeof(double) * 5*N(params.nx, params.ny));
  int* ja = (int*) malloc(sizeof(int) * 5*N(params.nx, params.ny));
  int* ia = (int*) malloc(sizeof(int) * (N(params.nx, params.ny) + 1));
  double* b = (double*) malloc(sizeof(double) * N(params.nx, params.ny));
  double* V = (double*) malloc(sizeof(double) * N(params.nx, params.ny));

  int nz_num = fill_matrix(params, ja, ia, a, b, matrix_file, vector_file);
  pmgmres_ilu_cr(N(params.nx, params.ny), nz_num, ia, ja, a, V, b, itr_max, mr, tol_abs, tol_rel);

  // zapisanie danych do mapy
	if (map_file){
		double tmp = 0.0;
		for(int z = 0; z < N(params.nx, params.ny); ++z){
			if(delta*j(params.nx,z) > tmp)
				fprintf(map_file, "\n");
			fprintf(map_file, "%f %f %f \n", delta*i(params.nx, z), delta*j(params.nx, z), V[z]);
			tmp = delta*j(params.nx, z);
		}
	}

  free(a);
  free(ja);
  free(ia);
  free(b);
  free(V);
}


int main(void) {
  struct Params params;
  initialize_params(&params);

  // 1. testowe wywołanie z zapisaniem macierzy i wektora do pliku
  FILE * matrix_file = fopen("matrix_test.dat", "w");
  FILE * vector_file = fopen("vector_test.dat", "w");
  solve_poisson_equation(params, NULL, matrix_file, vector_file);
  fclose(matrix_file);
  fclose(vector_file);

  // 2.
  // map1: nx = ny = 50
  set_node_numbers(&params, 50, 50);
	FILE * map_file = fopen("map1.dat", "w");
  solve_poisson_equation(params, map_file, NULL, NULL);
  fclose(map_file);

  // map2: nx = ny = 100
  set_node_numbers(&params, 100, 100);
	map_file = fopen("map2.dat", "w");
  solve_poisson_equation(params, map_file, NULL, NULL);
  fclose(map_file);

  // map3: nx = ny = 200
  set_node_numbers(&params, 200, 200);
	map_file = fopen("map3.dat", "w");
  solve_poisson_equation(params, map_file, NULL, NULL);
  fclose(map_file);

  // 3.
  // map4: eps1 = eps2 = 1.0, nx = ny = 100, V1 = V2 = V3 = V4 = 0.0
  set_node_numbers(&params, 100, 100);
	set_potentials(&params, 0.0, 0.0, 0.0, 0.0);
  set_max_ranges(&params, delta*params.nx, delta*params.nx);

	map_file = fopen("map4.dat", "w");
  solve_poisson_equation(params, map_file, NULL, NULL);
  fclose(map_file);

  // map5: eps1 = 1.0, eps2 = 2.0, nx = ny = 100
  set_epsilons(&params, 1.0, 2.0);
	map_file = fopen("map5.dat", "w");
  solve_poisson_equation(params, map_file, NULL, NULL);
  fclose(map_file);

  // map6: eps1 = 1.0, eps2 = 10.0, nx = ny = 100
  set_epsilons(&params, 1.0, 10.0);
	map_file = fopen("map6.dat", "w");
  solve_poisson_equation(params, map_file, NULL, NULL);
  fclose(map_file);

  return 0;
}