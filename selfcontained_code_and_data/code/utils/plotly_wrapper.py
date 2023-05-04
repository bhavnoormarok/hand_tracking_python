from utils.freq_imports import *


def mesh3d(v, F, intensity=None, colorscale=None, opacity=1, intensitymode='vertex', color=None, colorbar=None, customdata=None, hovertemplate=None, showscale=False, cmin=None, cmax=None, flatshading=False):
    """Wrapper for plotly.graph_objects.Mesh3d

    Args:
        color: string
            sets color of whole mesh
            Ref: https://developer.mozilla.org/en-US/docs/Web/CSS/color_value
        intensitymode: 'vertex' or 'cell'
            determines if intensity is specified for vertices or faces (also called cells in plotly)
        intensity: array-like 1D
            scalar field on vertices or faces
        colorscale: array containing arrays mapping a normalized value to an rgb, rgba, hex, hsl, hsv, or named color string
            e.g.Blackbody,Bluered,Blues,Cividis,Earth,Electric,Greens,Greys,Hot,Jet,Picnic,Portl and,Rainbow,RdBu,Reds,Viridis,YlGnBu,YlOrRd.
            Ref: https://plotly.com/python/builtin-colorscales/#builtin-sequential-color-scales
    """
    # use intensitymode='cell' if intensity specifies colors on triangles
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
        color=color, opacity=opacity, flatshading=flatshading, intensity=intensity, colorscale=colorscale, intensitymode=intensitymode, hoverinfo='none', showscale=showscale, colorbar=colorbar, customdata=customdata, hovertemplate=hovertemplate, cmin=cmin, cmax=cmax,
    )

def wireframe3d(v, F, opacity=0.5):
    verts_in_tris = v[F]
    Xe = []
    Ye = []
    Ze = []
    for verts_in_tri in verts_in_tris:
        Xe.extend([verts_in_tri[k % 3][0] for k in range(4)]+[None])
        Ye.extend([verts_in_tri[k % 3][1] for k in range(4)]+[None])
        Ze.extend([verts_in_tri[k % 3][2] for k in range(4)]+[None])
    wireframe = go.Scatter3d(
        x=Xe, y=Ye, z=Ze,
        mode="lines",
        line=dict(color='rgb(70,70,70)', width=1),
        marker=dict(
            opacity=opacity,
        ),
        hoverinfo='none',
        showlegend=False,
    )

    return wireframe


def line3d(start, end, color='red', width=2):
    return go.Scatter3d(
        x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
        mode='lines',
        line=dict(
            width=width,
            color=color,
        ),
        hoverinfo='none',
        showlegend=False
    )

def scatter3d(pts, size=None, color=None, colorscale=None, opacity=1, text=None, showlegend=False, hoverinfo=None, customdata=None, hovertemplate=None, colorbar=None, name=None):
    # set colorbar=dict(thickness=10) if colorbar needs to be shown
    scat3d = go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(
            size=size,
            color=color,
            colorscale=colorscale,
            opacity=opacity,
            colorbar=colorbar
        ),
        hoverinfo=hoverinfo,
        customdata=customdata,
        hovertemplate=hovertemplate,
        showlegend=showlegend,
        name=name,
    )

    if text is not None:
        scat3d.update(dict(hovertext=np.arange(pts.shape[0]) if isinstance(text, int) else text, hoverinfo='text'))

    return scat3d

def cone3d(xyz, uvw, size=5, opacity=0.5, colorscale=None):
    return go.Cone(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        u=uvw[:, 0], v=uvw[:, 1], w=uvw[:, 2],
        sizemode="absolute",
        sizeref=size,
        opacity=opacity,
        colorscale=colorscale,
        showscale=False,
    )


def scatter2d(pts, size=None, color=None, colorscale=None, opacity=1, text=None, showlegend=False, hoverinfo=None, customdata=None, hovertemplate=None, colorbar=None, mode="markers", name=None):
    scat2d = go.Scatter(
        x=pts[:, 0], y=pts[:, 1],
        mode=mode,
        marker=dict(
            size=size,
            color=color,
            colorscale=colorscale,
            opacity=opacity,
            colorbar=colorbar
        ),
        hoverinfo=hoverinfo,
        customdata=customdata,
        hovertemplate=hovertemplate,
        showlegend=showlegend,
        name=name,
    )

    if text is not None:
        scat2d.update(dict(hovertext=np.arange(pts.shape[0]) if isinstance(text, int) else text, hoverinfo='text'))

    return scat2d

def coordinate_frame(origin, offset_x, offset_y, offset_z):
    xaxis = go.Scatter3d(x=[origin[0], origin[0]+offset_x[0]], y=[origin[1], origin[1]+offset_x[1]], z=[origin[2], origin[2]+offset_x[2]],
        mode='lines',
        line=dict(
            width=2,
            color='red',
        ),
        hoverinfo='none',
        showlegend=False
    )
    yaxis = go.Scatter3d(x=[origin[0], origin[0]+offset_y[0]], y=[origin[1], origin[1]+offset_y[1]], z=[origin[2], origin[2]+offset_y[2]],
        mode='lines',
        line=dict(
            width=2,
            color='green',
        ),
        hoverinfo='none',
        showlegend=False
    )
    zaxis = go.Scatter3d(x=[origin[0], origin[0]+offset_z[0]], y=[origin[1], origin[1]+offset_z[1]], z=[origin[2], origin[2]+offset_z[2]],
        mode='lines',
        line=dict(
            width=2,
            color='blue',
        ),
        hoverinfo='none',
        showlegend=False
    )
    origin_marker = go.Scatter3d(x=[origin[0]], y=[origin[1]], z=[origin[2]],
        mode='markers',
        marker=dict(
            size=2,
            color='black',
            opacity=0.8,
        ),
        hoverinfo='none',
        showlegend=False
    )
    return xaxis, yaxis, zaxis, origin_marker

def line3d(start, end, color='red', width=2):
    return go.Scatter3d(
        x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
        mode='lines',
        line=dict(
            width=width,
            color=color,
        ),
        hoverinfo='none',
        showlegend=False
    )

def line2d(start, end, color='red', width=2):
    return go.Scatter(
        x=[start[0], end[0]], y=[start[1], end[1]],
        mode='lines',
        line=dict(
            width=width,
            color=color,
        ),
        hoverinfo='none',
        showlegend=False
    )

def draw_skeleton_lines(k, color='darkgray', width=10):
    lines = []
    wrist_id = 0
    for i_f in range(5):
        mcp_id = 4*i_f+1; pip_id = mcp_id+1; dip_id = pip_id+1; tip_id = dip_id + 1
        lines.append(line3d(k[wrist_id], k[mcp_id], color=color, width=width))
        lines.append(line3d(k[mcp_id], k[pip_id], color=color, width=width))
        lines.append(line3d(k[pip_id], k[dip_id], color=color, width=width))
        lines.append(line3d(k[dip_id], k[tip_id], color=color, width=width))
    return lines


def remove_fig_background(fig):
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False,
                       title="", visible=False),
            yaxis=dict(showbackground=False, showticklabels=False,
                       title="", visible=False),
            zaxis=dict(showbackground=False, showticklabels=False,
                       title="", visible=False),
        )
    )

def remove_2d_axes(fig):
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)


def update_fig_size(fig, width=500, height=500, l=10, r=10, t=10, b=10, pad=1):
    fig.update_layout(
        # autosize=False,
        width=width,
        height=height,
        margin=dict(
            l=l,
            r=r,
            b=b,
            t=t,
            pad=pad
        ),
    )

def invert_fig_y(fig):
    # fig.update_layout(yaxis=dict(autorange='reversed'))
    fig.update_yaxes(autorange='reversed')

def fit_fig_to_shape(fig, shape):
    fig.update_layout(width=shape[1], height=shape[0])

def update_fig_camera(fig, up=[0, -1, 0], center=[0, 0, 0], eye=[0, 0, -3]):
    camera = dict(
        up=dict(x=up[0], y=up[1], z=up[2]),    # up in page
        center=dict(x=center[0], y=center[1], z=center[2]), # projection of center point lies at center of view
        eye=dict(x=eye[0], y=eye[1], z=eye[2])  # view point (position of camera) and zoom
    )
    fig.layout.scene.camera = camera
