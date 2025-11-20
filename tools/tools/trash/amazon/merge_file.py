

import os
import shutil
source_dir = "source/assets/ycb/dexgrasp/original_mesh"
target_dir = "source/assets/ycb/dexgrasp/raw_mesh"
object_list = os.listdir(source_dir)

for object_name in object_list:
   
    object_file = f"{source_dir}/{object_name}/textured.obj"
  
    change_file_name = "_".join(object_name.split("_")[1:])
    change_dir_name = f"{target_dir}/{change_file_name}.obj"
    shutil.copy(object_file, change_dir_name)
