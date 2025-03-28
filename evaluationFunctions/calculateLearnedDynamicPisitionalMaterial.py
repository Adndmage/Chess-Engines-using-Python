"""
identical to calculateDynamicPisitionalMaterial.py but have learnable values for positional value
"""
# calculate numpy bitboard for each piecetype
# multiply by positional piece value (ie by another numpy bitboard instead of a scalar)
# sum all piece values for each side
# return difference between white and black


# add ai training functionality to change the weights of the positional piece value bitboards 