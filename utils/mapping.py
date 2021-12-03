

'''
---  S T A R T  O F  F U N C T I O N  F E A T U R E _ S C A L I N G ---
    [About]
        Function for scaling input `xs` from distribution `source_dist` to a distribution `target_dist`.
    [Args]
        - xs: Float or Integer that should fall within `source_dist`
        - source_min: Float or Integer for the minimum allowed value at the source distribution
        - source_max: Float or Integer for the maximum allowed value at the source distribution
        - target_min: Float or Integer for the minimum allowed value at the target distribution
        - target_max: Float or Integer for the maximum allowed value at the target distribution.
    [Returns]
        - xt: Float or Integer that has been re-schaled.
'''
def feature_rescaling(xs,source_min,source_max,target_min,target_max):
    xt = target_min + ((xs-source_min)*(target_min-target_max)/(source_max-source_max))
    return xt
'''
---  E N D  O F  F U N C T I O N  F E A T U R E _ S C A L I N G ---
'''
