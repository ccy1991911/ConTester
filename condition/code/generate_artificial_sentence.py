timer_list = ['T3402', 'T3410', 'T3411', 'T3412', 'T3416', 'T3417', 'T3418', 'T3420', 'T3421', 'T3423', 'T3430', 'T3440', 'T3442', 'T3444', 'T3445', 'T3447', 'T3448', 'T3449', 'T3413', 'T3415', 'T3422', 'T3447', 'T3450', 'T3460', 'T3470', 'T3346', 'T3247']

d = 6807
for t in timer_list:
    print('\n{%d, 0} If UE starts timer %s, the timer %s will expire after the period of time as specified by %s.\n---'%(d, t, t, t))
    d = d + 1
    print('\n{%d, 0} If UE starts timer %s, the timer %s will expire.\n---'%(d, t, t))
    d = d + 1

