#pragma once
#ifdef __CUDACC__
#define CHD __host__ __device__
#else
#define CHD
#endif
CHD inline int cheb_leaf(const double* tree,double a,double beta,double old,double exponent){
 double x[3]={2*(old/(1+beta)+beta/(1+beta)*a)-1,2*log1p(beta)/log1p(2500000.)-1,2*(exponent-.2)/1.8-1};
 int node=0;const double* r=tree;
 while(r[0]>=0){node=int(x[int(r[0])]<=r[3]?r[1]:r[2]);r=tree+74*node;}
 return node;
}
CHD inline void cheb_value(const double* tree,double a,double beta,double old,double exponent,double& value,double& deriv,int leaf_override=-1){
 double x[3]={2*(old/(1+beta)+beta/(1+beta)*a)-1,2*log1p(beta)/log1p(2500000.)-1,2*(exponent-.2)/1.8-1};
 int node=leaf_override<0?0:leaf_override;const double* r=tree+74*node;
 while(r[0]>=0){node=int(x[int(r[0])]<=r[3]?r[1]:r[2]);r=tree+74*node;}
 double v[3][4],dv[4];
 for(int j=0;j<3;j++){double z=(x[j]-r[4+j])*r[7+j]-1;v[j][0]=1;v[j][1]=z;v[j][2]=2*z*z-1;v[j][3]=4*z*z*z-3*z;if(j==0){dv[0]=0;dv[1]=1;dv[2]=4*z;dv[3]=12*z*z-3;}}
 value=deriv=0;
 for(int i=0;i<4;i++)for(int j=0;j<4;j++)for(int k=0;k<4;k++){double c=r[10+16*i+4*j+k];value+=c*v[0][i]*v[1][j]*v[2][k];deriv+=c*dv[i]*v[1][j]*v[2][k];}
 deriv*=r[7]*2*beta/(1+beta);
}
