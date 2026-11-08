#include "nvm2c_shape.h"
#include <stdio.h>
int main(void) {
 NvmShapeGraph g={0};
 NvmShapeId producer=nvm_shape_new(&g,NVM_SHAPE_MAP);
 NvmShapeId checked=nvm_shape_new(&g,NVM_SHAPE_MAP);
 NvmShapeId element=nvm_shape_new(&g,NVM_SHAPE_INT);
 NvmShapeId read=nvm_shape_new(&g,NVM_SHAPE_UNKNOWN);
 nvm_shape_unify(&g,nvm_shape_child(&g,producer,0),element);
 nvm_shape_unify(&g,nvm_shape_child(&g,producer,1),element);
 nvm_shape_array_read(&g,element,read);
 nvm_shape_unify(&g,nvm_shape_child(&g,checked,0),read);
 nvm_shape_convert(&g,producer,checked);
 int ok=nvm_shape_solve_conversions(&g);
 printf("I observed %s: %s\n",ok?"acceptance":"refusal",g.error?g.error:"none");
 nvm_shape_destroy(&g);return ok;
}
