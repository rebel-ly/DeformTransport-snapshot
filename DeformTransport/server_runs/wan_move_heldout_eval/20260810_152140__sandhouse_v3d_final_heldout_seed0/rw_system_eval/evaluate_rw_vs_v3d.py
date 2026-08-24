from pathlib import Path
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from torchvision.models.optical_flow import (
    raft_large,
    Raft_Large_Weights,
)


DT=Path("/workspace/DeformTransport")

RUN=(
    DT
    / "server_runs/wan_move_heldout_eval/"
      "20260810_152140__sandhouse_v3d_final_heldout_seed0/"
      "rw_system_eval"
)

FORMAL=(
    DT
    / "server_runs/wan_move_formal/"
      "20260810_124701__v3d_mechanism_and_sandhouse_heldout_seed0"
)

RW_NPY=(
    RUN
    / "realwonder_states_0_2_160_rgb_uint8.npy"
)

CORRECT=(
    FORMAL
    / "sandhouse_correct/"
      "sandhouse_v3d_correct_seed0.mp4"
)

TRACKS=(
    FORMAL
    / "artifacts/sandhouse/"
      "sandhouse_v3d_correct_tracks.npy"
)

VIS=(
    FORMAL
    / "artifacts/sandhouse/"
      "sandhouse_v3d_visibility.
)

SOURCE=

    / "server_runs/sand_
      "v1_frozen/20260808_2

)


ANCHORS=list(

)

OFF=np.arange(
    -3.5,
    4.0,
    1.0,
    dtype=np.float3
)

BOOT_N=
BOOT_SEED=


# Existing f

EXPECTED_V3D_TCMAR=23.2



d
    diff,


):
    d=np
        diff,
        n
    )

    n=l

    rng=np.r
        se
    )

    vals=[]

 

        nboot

    ):
        k=m
            500,
   
        )

   
      
            n
         
 

        vals.appe
     
                axis=1
            )
   

    vals=n
 
  

    ret
        float(
            np
                vals
             
        
        ),
        float(
            np.percenti
      
    

        ),
    ]


def rea
    bgr=cv2.imread(
        str(SOURCE),
        cv2.IMREAD_COLOR
    )

    if bgr is 
        rai
  


   
        480,
        832,

    )

    return (
        cv2.cvtColor(
            bgr,


    
        )
        /255.0
    )


def to_common_batch(
    rgb_u8
):
    assert (
        rgb_u8.ndim == 4
        and rgb_u8
    )

    h,w=rgb_u8.shape[1:3]

    if (h,w) == (464,832):

     

                np.float32
            )
      
        )

    if (h,w

        raise
            (h,w)
      

    out=[]

    for s in rang

        len(rgb_u8),

    ):
        x=torc

        ).permute(
           


        

           
            mode="bic
            al
            a

            0,
            1
        )

      

                0,2,3
            
        )

    return np.
     

    ).astype(
        



def read_co

        str(CORRECT)
    )

    frames

    while True:
        ok,b

        if not ok:
 

     
        
      
                cv
 


    cap.release()

    ass

    arr=np.stack(
     
        axis=0
    )

    a

        464,
      

    ), arr.shape

    return (
   
            np.float32
     

    )



    img,
    centers
):
    H,W=img.shape[:2]

   
       
 


    xs=
        centers[
            :,
            
            N
 

        +
        

            None,
       
        ]
    )

 
        centers[
            
           
      
            None
  

        
            None,
 
            None



    xs=np.broadcast
        xs,
        (


            8
        )
   

    ys=np.broadcast_
        ys,
        (
            len(cen
      

        )
    )

    vali
        (
            xs.mi

            ) >= 0
      
        &
        (

                (1,2)
            ) >= 0
 
        &
        (
    
                (1,2)
      

        &
        (
       
                (1,2)
            ) <= H-1
        )
    )

    if not np.all(
        valid
    ):
        raise RuntimeError(
            "invali
 

 
        xs
    ).astype(
        np.in
    )

    y0
        ys
    
        np.int6
    )

    x1=np
        x0
 


    y1=np.minimum(
        y0+1,
        H-1
    )

    wx=(
    
    )[...,None]

    wy=(
        ys-y0
 

    Ia=img[y0,x0]
    Ib=img[y0,x1]
    Ic=img[y1,x0]
    Id=img[y1,x1]

  
  


    
            +
            (1-wx)*wy*Ic
            +
            wx
        )
        .astype(
            np.float32
 
    )


def patch_mean_lab(
    patches
):
    n=patches.shape[0]



        8,
        3
    ).astype(
        np.float32
  

    lab=cv2.cvtColor(
 
        cv2
    ).res

        8,
        8,
       
    )

    r
        axis=(1,2)
    )


def 

    ntracks,
):
    sums=np.zeros(
        nt
        np.floa
    )

    cnt=np.zeros
 



    for ids,vals in rows:

        n

     
      
        )

        np.add.at(
            
            ids,
            1
        )

    out=np.full(

        np.nan,
      
 

    m=cn

    o
        sums[m]
        /
        cn
    )

    return out,cnt


def tc_mar(
    tracks,
    vis,
    r
 

    sour

    n=

    src_centers=tracks[0]

    src_valid=(
   
 

        )
        &

            src_centers[:,
            <=831
        )
        &
 
 

        
        &

            src_centers[:,1]
            <=479
        )
        &
  
 

     
        )


    good=np.where(
        
    )[0]

    src_patch=np.full(
 

       

            3
        ),
        np.nan,
        np.





        sourc

 

    )

    src
        (
        
        
        ),
        
 


    sr
        go


            

    )

    rows={
     



 

    for 


  
        

        # Fr
        c
     


      

            
          
     

           

 

             

      
           
                
   
         

   
      
             
         
 

        valid=(
 
     
            src_valid

            f
   

        id
 
  

       
            id
        )

   
            ("rw",rw
            (
        

         
              
                c[ids]


     
    


         
     


       
            )

    
                name
            ].append(
  
     

              
           

 


   

        pt[
            

     

    assert n
        counts["rw"],
        counts["


    
        c
    )

    val
     


    # Reproduce froz
    assert
  

    rw_mean
        pt["rw"][
      
        ].mean()
 

    

            valid
       


    diff=(
        pt["rw

     

        pt["v3d"][
       
        ]
   

    c
        d


    decisi

        if ci
        else (
  
      

        )


    return {
    


        "lower_bett


     
            rw

        "v3d":
   

        "r


        

        "bo
            ci,

    
            de

        "boo


        "vali
            v

        

    }


def load_raft(
    de
):
    weigh
        R

    )

    bas
     


    cache=(

        



        /
 


    print(
        
     

    )

   

        raise R
            

        )

    mod
 

    )
        
    )

    return (
     
 



def infer_flows(

    mod

    device,
    b
):
  

    with torc

    

     

        ):
 
      

            )

 

            
   
            ).to(
    
     

     



        
           
  

            a=F.inte

   
       
 


       
                
              
            

            
 

         

       

                b
       

        
     



                
            
           

    r
        outs,
  
  


def bil
    flow,
    cen
)
    H,W=flow.sha



        np.float32


    x=c[:,
    y=c[:


        x
   
        n
   

    y0=np.floor(
   
    ).astyp
        n
    )

    x1=np.mi
      

    )

  
     

    )

 
    wy=y-

    f=flow.trans

    )

    Ia=f[y0
    Ib
    Ic=f[
    Id=f[

    return (
        
        *
        
 
        I
        +
    
        *
        (1-
      

        +
        (
       
        wy[:,None]
  
        Ic
        +
        w
     

        *
        
    )


def t
    tr
    vis,
    rw,
    correc
):
    device=torch
 



        de
    )

    pr
        "RAFT
     


    r
        rw
    
        transfo
     

    )

  
        "R
 


    v3d_flow=infer
        corre
        mod
     

        
    

    del model


    per=
        "rw":
 


    for t in ran
        80
    ):

        valid=(

            &
   

  
  


    
            &
            np.isfinite(
             
            ).
         
            )
  

        ids=np.where(
 
     


        centers=(
 
           
  
            ]
        



          
         
             
            ]
    
  

                ids
 
 

        in
         

          
          
       
     

     
            (
    
     


    

        )

 
  
        ]

       
          
        ]

    
     


            rai
 



            ("rw",rw_flow

        ]

     
      
         


            epe=n
            
                
            )

        

            ].ap

               
      
 

    rw=n

     
    )

    v3d=
        p
        np
    )

    diff=rw-v3d

 



    decisi
        "V3
        
     
 

        

    )


        "metric":
       

        "lower_
   



         
         

        "v3d":
           
                v
         

        
 
 


       
         

        "decision":
        

        "bootstr
         

        
  
 


def 
    track


        np.float32
    )

 
        

        bool
    )

  
 

       


    assert v
        81
        1791
  

    rw_raw





        480,


 

    a

    pri
        "
        
    )

 
        rw
    )

 
 



    c

    asser


        832,

    )

    print(
     



 

        


  

    pri

            
         
     



    #

        appe
        -

    )

        rai

 

        )

  

      
    )

    
        "RUN TC-
   
    )

  

   
      
        corre
    )

  
 

            ind
 
     
    )

    # Small GP

        motio
   

    ) > 5e


  

       
        )

   
        "

   

    report={
      
            "

       


        
            "r
                165,




     
    


         

    


       
             

    
                ),


                "832x

 
     

            "v
           





   

    }

    
        RUN


     

            
        )
        + "
    )

    print


    

    prin
    p

       
     


        "V3D        
        ap
  

        "RW
        appearanc
    )

        "95% CI 
 

    

        appearanc
    )




        "R
        motion

    p

        motion["v3
    )
 
        "
   

    p
        "


    print(

        motio
    )

    pri
  
      

    print


if __name__ 
    



"$PY" -m py_compil




ech
echo "========

echo "========

GP

import sub


    "nvi

    "--form
],text=True)

r

for 
    i,free,uti

        for 


        (i,fr
    )

for wa

        

    )


        print(
      

r
    "No s
)
