import mplcyberpunk


def enhance_plot(figure, axes, glow=False, alpha_gradient=0, lines=True):
    figure.set_facecolor('black')
    axes.set_facecolor('black')
    if glow:
        if lines:
            mplcyberpunk.make_lines_glow(ax=axes)
        else:
            mplcyberpunk.make_scatter_glow(ax=axes)
    if 1 > alpha_gradient > 0:
        mplcyberpunk.add_gradient_fill(ax=axes, alpha_gradientglow=alpha_gradient)