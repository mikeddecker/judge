def get_discipline_DoubleDutch_config():
    return {
        "Tablename" : "DoubleDutch",
        "Type" : ("Categorical", "Type"), # Will be textual representions
        "Rotations" : ("Numerical", 0, 8, 1), # min, max, step
        "Turner1": ("Categorical", "Turner"),
        "Turner2": ("Categorical", "Turner"),
        "Skill" : ("Categorical", "Skill"),
        "Hands" : ("Numerical", 0, 2, 1), # 0 for al salto types
        "Feet" : ("Numerical", 0, 2, 1),
        "Turntable" : ("Numerical", 0, 4, 0.25), # Per quarter (but still integers)
        "BodyRotations" : ("Numerical", 0, 2, 0.5),
        "Backwards" : ("Boolean"),
        "Sloppy" : ("Boolean"),
        "Hard2see" : ("Boolean"),
        "Fault" : ("Boolean"),
    }