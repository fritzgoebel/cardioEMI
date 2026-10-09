// app-ws-preview.js - live 3D preview of one weak-scaling cell.
//
// The cell is fully analytic (a cylinder along x plus two oblique tubes), so the
// preview needs no server round-trip: it mirrors geometry/cell_box_gmsh.py and
// generate_weak_scaling_mesh.cell_contains exactly. Neighbour cells are drawn
// faintly at the lattice vectors a1 = (Lx,0,0) and a2 = (shift,L,0), which is
// what makes the shifted lattice legible -- you can see a connector leave one
// cell and arrive in the box above.

const WS_SEG = 48;

App.prototype.initWeakScalingPreview = function() {
    // The preview lives in the main viewer as an overlay layer, so it gets the
    // full canvas and the same navigation as a real mesh. It keeps its own
    // scene rather than borrowing the Viewer's, because Viewer.init needs
    // meshData -- and the whole point is to preview before anything exists.
    const host = document.getElementById('ws-preview-layer');
    if (!host || !window.THREE) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a1a);      // match the mesh viewer
    const camera = new THREE.PerspectiveCamera(38, 1, 1, 20000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    host.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 0.75);
    key.position.set(1, 1.4, 1.2);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x88aaff, 0.35);
    fill.position.set(-1, -0.6, -0.8);
    scene.add(fill);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;

    const group = new THREE.Group();
    scene.add(group);

    this._wsPreview = { scene, camera, renderer, controls, group, host, sized: 0 };

    const tick = () => {
        requestAnimationFrame(tick);
        if (host.style.display === 'none') return;     // idle while hidden
        const w = host.clientWidth, h = host.clientHeight;
        if (w && h && (w !== this._wsPreview.sized || h !== this._wsPreview.sizedH)) {
            // No third argument: the canvas CSS size must track the buffer, or
            // it stays at the default 300x150 and the render is cropped into
            // the corner of the layer.
            renderer.setSize(w, h);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            this._wsPreview.sized = w;
            this._wsPreview.sizedH = h;
            this.frameWeakScalingPreview();   // the fit depends on the aspect
        }
        controls.update();
        renderer.render(scene, camera);
    };
    tick();
    this.updateWeakScalingPreview();
};

// A disc of `radius` at `centre` whose plane has the given axis normal ('y'|'z'),
// swept along `vec`. Sweeping a disc -- rather than sweeping a cylinder along a
// tilted axis -- is what keeps every cross-section parallel to the face an exact
// circle, so this is the same construction the mesher uses.
function wsObliqueTube(centre, normal, vec, radius) {
    const u = normal === 'y' ? [1, 0, 0] : [1, 0, 0];
    const v = normal === 'y' ? [0, 0, 1] : [0, 1, 0];
    const pos = [], idx = [];
    for (let i = 0; i <= WS_SEG; i++) {
        const t = (i / WS_SEG) * Math.PI * 2;
        const c = Math.cos(t) * radius, s = Math.sin(t) * radius;
        const p = [centre[0] + u[0] * c + v[0] * s,
                   centre[1] + u[1] * c + v[1] * s,
                   centre[2] + u[2] * c + v[2] * s];
        pos.push(p[0], p[1], p[2]);
        pos.push(p[0] + vec[0], p[1] + vec[1], p[2] + vec[2]);
    }
    for (let i = 0; i < WS_SEG; i++) {
        const a = 2 * i, b = 2 * i + 1, c = 2 * i + 2, d = 2 * i + 3;
        idx.push(a, c, b, b, c, d);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
}

function wsBody(radius, Lx, yc, zc) {
    const g = new THREE.CylinderGeometry(radius, radius, Lx, WS_SEG, 1, false);
    g.rotateZ(Math.PI / 2);                   // the cylinder's axis is +y by default
    g.translate(Lx / 2, yc, zc);
    return g;
}

App.prototype.buildWeakScalingCell = function(p, material) {
    const L = p.L, ax = p.shape === 'plus' ? 1 : p.ax;
    const Lx = ax * L, c = L / 2;
    const R = 0.38 * L;                       // CellShape body_r default
    const rl = (p.lat_r || 0.26) * L;
    const dY = (p.d_y || 0.5) * L, dZ = (p.d_z || p.d_y || 0.5) * L;
    // A stub travels this far in x on its way from its face to the body axis.
    const lean = (0.5 / Math.tan((p.lean || 55) * Math.PI / 180)) * L;

    // Four stubs, not two tubes through the cell: each direction has one near
    // the small-x face and one near the large-x face, both the same distance in,
    // and both leaning the same way. That distance is what sets the shift.
    const cell = new THREE.Group();
    cell.add(new THREE.Mesh(wsBody(R, Lx, c, c), material));
    for (const [centre, nrm, vec] of [
            [[dY, 0, c], 'y', [lean, c, 0]],
            [[Lx - dY, L, c], 'y', [-lean, -c, 0]],
            [[dZ, c, 0], 'z', [lean, 0, c]],
            [[Lx - dZ, c, L], 'z', [-lean, 0, -c]]]) {
        cell.add(new THREE.Mesh(wsObliqueTube(centre, nrm, vec, rl), material));
    }
    return cell;
};

App.prototype.updateWeakScalingPreview = function() {
    const pv = this._wsPreview;
    if (!pv) return;
    // _wsLastParams carries the resolved angle/width/positions; the raw read is
    // only a fallback for the very first frame.
    const p = this._wsLastParams || this.readWeakScalingParams();
    if (this.weakScalingValidation(p)) return;      // leave the last good frame up

    while (pv.group.children.length) pv.group.remove(pv.group.children[0]);

    const L = p.L, ax = p.shape === 'plus' ? 1 : p.ax;
    const slabs = p.slabs || 2, step = p.step_y || 1, stepZ = p.step_z || step;
    const Lx = ax * L;
    const shiftY = (ax * step / slabs) * L, shiftZ = (ax * stepZ / slabs) * L;

    const solid = new THREE.MeshPhongMaterial({
        color: 0xe8845c, shininess: 25, side: THREE.DoubleSide });
    const ghost = new THREE.MeshPhongMaterial({
        color: 0x6f8fb5, transparent: true, opacity: 0.22,
        side: THREE.DoubleSide, depthWrite: false });

    pv.group.add(this.buildWeakScalingCell(p, solid));
    // Neighbours at +-a1 and +-a2: the x ones butt end to end, the y ones sit
    // `shift` further along x, which is where this cell's connector arrives.
    // Both signs, so the composition stays symmetric about the cell and the
    // cell really is centred rather than merely targeted.
    for (const off of [[Lx, 0, 0], [-Lx, 0, 0],
                       [shiftY, L, 0], [-shiftY, -L, 0],
                       [shiftZ, 0, L], [-shiftZ, 0, -L]]) {
        const g = this.buildWeakScalingCell(p, ghost);
        g.position.set(off[0], off[1], off[2]);
        pv.group.add(g);
    }

    const box = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(Lx, L, L)),
        new THREE.LineBasicMaterial({ color: 0x9fb4cc, transparent: true, opacity: 0.5 }));
    box.position.set(Lx / 2, L / 2, L / 2);
    pv.group.add(box);

    // Frame the cell's own box, not the ghosts: the ghosts are context and may
    // bleed off the edges. Only refit when the box actually changes size, so
    // dragging the interface slider does not yank the camera back each frame.
    pv.centre = new THREE.Vector3(Lx / 2, L / 2, L / 2);
    pv.radius = 0.5 * Math.sqrt(Lx * Lx + 2 * L * L) * 1.12;
    const sig = `${Lx}:${L}`;
    if (sig !== pv.fitted) {
        pv.fitted = sig;
        this.frameWeakScalingPreview();
    }
};

// Point the camera at the cell and pull back far enough that its bounding
// sphere fits both the vertical and the horizontal field of view -- fitting the
// vertical alone leaves a long box overflowing a wide canvas. Keeps whatever
// direction the user has orbited to.
App.prototype.frameWeakScalingPreview = function() {
    const pv = this._wsPreview;
    if (!pv || !pv.centre) return;
    const cam = pv.camera;
    const vfov = cam.fov * Math.PI / 180;
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * (cam.aspect || 1));
    const dist = Math.max(pv.radius / Math.sin(vfov / 2),
                          pv.radius / Math.sin(hfov / 2));

    let dir = new THREE.Vector3().subVectors(cam.position, pv.controls.target);
    if (dir.lengthSq() < 1e-9) dir.set(0.55, 0.45, 1.0);
    dir.normalize();

    cam.position.copy(pv.centre).addScaledVector(dir, dist);
    cam.near = Math.max(dist / 1000, 0.01);
    cam.far = dist * 100;
    cam.updateProjectionMatrix();
    pv.controls.target.copy(pv.centre);
    pv.controls.update();
};


// Show or hide the preview layer. Hiding it also reveals whatever the mesh
// viewer is drawing underneath, so switching tabs restores the real mesh.
App.prototype.showWeakScalingPreview = function(on) {
    const host = document.getElementById('ws-preview-layer');
    if (!host) return;
    host.style.display = on ? 'block' : 'none';
    if (on) {
        // The layer had no size while hidden, so the renderer must re-fit.
        if (this._wsPreview) {
            this._wsPreview.sized = 0;
            this._wsPreview.fitted = null;    // re-frame at the real aspect
        }
        this.updateWeakScalingPreview();
    }
};
