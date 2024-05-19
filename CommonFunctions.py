import mplcyberpunk


def enhance_plot(figure, axes, glow=False, alpha_gradient=0, lines=True, dpi=100):
    figure.set_facecolor('black')
    figure.set_dpi(dpi)
    axes.set_facecolor('black')
    for font in [axes.title, axes.xaxis.label, axes.yaxis.label]:
        font.set_fontweight('bold')
    if glow:
        if lines:
            mplcyberpunk.make_lines_glow(ax=axes)
        else:
            mplcyberpunk.make_scatter_glow(ax=axes)
    if 1 > alpha_gradient > 0:
        mplcyberpunk.add_gradient_fill(ax=axes, alpha_gradientglow=alpha_gradient)