import pinocchio

filename = "./assets/urdf/piper_description.urdf"
model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(filename)
