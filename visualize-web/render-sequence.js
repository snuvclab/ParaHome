import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { loadTxt, loadJSON, convertRgbToHex } from "./utils.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

// CONSTANTS
const FPS = 30;
const DEFAULT_ASPECT_RATIO = 16 / 9;
const BODY_COLOR = 0xddddff;
const HAND_COLOR = 0xffffdd;
const HAND_CYLINDER_RADIUS = 0.006;
const HAND_CYLINDER_DATA = [
  // left upper1 finger
  { start_joint: 23, end_joint: 24 },
  { start_joint: 23, end_joint: 25 },
  { start_joint: 23, end_joint: 26 },
  { start_joint: 23, end_joint: 27 },
  { start_joint: 23, end_joint: 28 },
  // left last finger
  { start_joint: 28, end_joint: 29 },
  { start_joint: 29, end_joint: 30 },
  { start_joint: 30, end_joint: 31 },
  { start_joint: 31, end_joint: 32 },
  // left forth finger
  { start_joint: 27, end_joint: 33 },
  { start_joint: 33, end_joint: 34 },
  { start_joint: 34, end_joint: 35 },
  { start_joint: 35, end_joint: 36 },
  // left third finger
  { start_joint: 26, end_joint: 37 },
  { start_joint: 37, end_joint: 38 },
  { start_joint: 38, end_joint: 39 },
  { start_joint: 39, end_joint: 40 },
  // left second finger
  { start_joint: 25, end_joint: 41 },
  { start_joint: 41, end_joint: 42 },
  { start_joint: 42, end_joint: 43 },
  { start_joint: 43, end_joint: 44 },
  // left first finger
  { start_joint: 24, end_joint: 45 },
  { start_joint: 45, end_joint: 46 },
  { start_joint: 46, end_joint: 47 },
  // right hand
  { start_joint: 48, end_joint: 49 },
  { start_joint: 48, end_joint: 50 },
  { start_joint: 48, end_joint: 51 },
  { start_joint: 48, end_joint: 52 },
  { start_joint: 48, end_joint: 53 },
  { start_joint: 53, end_joint: 54 },
  { start_joint: 54, end_joint: 55 },
  { start_joint: 55, end_joint: 56 },
  { start_joint: 56, end_joint: 57 },
  { start_joint: 52, end_joint: 58 },
  { start_joint: 58, end_joint: 59 },
  { start_joint: 59, end_joint: 60 },
  { start_joint: 60, end_joint: 61 },
  { start_joint: 51, end_joint: 62 },
  { start_joint: 62, end_joint: 63 },
  { start_joint: 63, end_joint: 64 },
  { start_joint: 64, end_joint: 65 },
  { start_joint: 50, end_joint: 66 },
  { start_joint: 66, end_joint: 67 },
  { start_joint: 67, end_joint: 68 },
  { start_joint: 68, end_joint: 69 },
  { start_joint: 49, end_joint: 70 },
  { start_joint: 70, end_joint: 71 },
  { start_joint: 71, end_joint: 72 },
];
const BODY_CYLINDER_DATA = [
  // feet
  { start_joint: 22, end_joint: 21, start_radius: 0.02, end_radius: 0.02 },
  { start_joint: 18, end_joint: 17, start_radius: 0.02, end_radius: 0.02 },
  // calf
  { start_joint: 21, end_joint: 20, start_radius: 0.02, end_radius: 0.03 },
  { start_joint: 17, end_joint: 16, start_radius: 0.02, end_radius: 0.03 },
  // thighs
  { start_joint: 20, end_joint: 19, start_radius: 0.03, end_radius: 0.04 },
  { start_joint: 16, end_joint: 15, start_radius: 0.03, end_radius: 0.04 },
  // forearm
  { start_joint: 14, end_joint: 13, start_radius: 0.02, end_radius: 0.02 },
  { start_joint: 10, end_joint: 9, start_radius: 0.02, end_radius: 0.02 },
  // upper arm
  { start_joint: 13, end_joint: 12, start_radius: 0.02, end_radius: 0.03 },
  { start_joint: 9, end_joint: 8, start_radius: 0.02, end_radius: 0.03 },
  // shoulder
  { start_joint: 12, end_joint: 11, start_radius: 0.03, end_radius: 0.02 },
  { start_joint: 8, end_joint: 7, start_radius: 0.03, end_radius: 0.02 },
  // spine
  { start_joint: 5, end_joint: 4, start_radius: 0.02, end_radius: 0.02 },
  { start_joint: 4, end_joint: 3, start_radius: 0.02, end_radius: 0.02 },
  { start_joint: 3, end_joint: 2, start_radius: 0.02, end_radius: 0.02 },
  { start_joint: 2, end_joint: 1, start_radius: 0.02, end_radius: 0.02 },
  { start_joint: 1, end_joint: 0, start_radius: 0.02, end_radius: 0.02 },
  // pelvis
  { start_joint: 19, end_joint: 0, start_radius: 0.02, end_radius: 0.02 },
  { start_joint: 15, end_joint: 0, start_radius: 0.02, end_radius: 0.02 },
];

const DEFAULT_CAMERA_POSITION = [2, 2, 3];
const DEFAULT_CAMERA_UP = [0, 0, 1];
const DEFAULT_LOOK_AT = [0, 0, 0];
const DEFAULT_FOCAL_LENGTH = 8;

// Utils
function getJointRadius(index) {
  // head
  if (index === 6) return 0.07;
  // hand
  if (index >= 23) return 0.006;
  // spine
  if (index >= 1 && index <= 4) return 0.03;
  // shoulder
  if (index === 12 || index === 8) return 0.035;

  // pelvis
  if (index === 19 || index === 15) return 0.04;

  // body
  return 0.02;
}

const cachedObjsByUrl = {};

export function loadObj(url) {
  return new Promise((resolve, reject) => {
    const cachedObj = cachedObjsByUrl[url];

    if (cachedObj) return resolve(cachedObj.clone());

    // instantiate a loader
    const loader = new OBJLoader();
    loader.load(
      url,
      (object) => {
        cachedObjsByUrl[url] = object;
        resolve(object.clone());
      },
      undefined,
      (error) => {
        if (error.message.includes("404")) {
          resolve(null);
          return;
        }
        console.error("Failed to load OBJ file");
        console.error(error);
        reject(error);
      }
    );
  });
}

function getCylinderLength(startJointPosition, endJointPosition) {
  const direction = new THREE.Vector3().subVectors(
    endJointPosition,
    startJointPosition
  );

  return direction.length();
}

function updateCylinderPosition(
  cylinder,
  startJointPosition,
  endJointPosition
) {
  const direction = new THREE.Vector3().subVectors(
    startJointPosition,
    endJointPosition
  );
  const axis = new THREE.Vector3(0, 1, 0);
  const quaternion = new THREE.Quaternion().setFromUnitVectors(
    axis,
    direction.clone().normalize()
  );
  cylinder.position.copy(
    new THREE.Vector3()
      .addVectors(startJointPosition, endJointPosition)
      .multiplyScalar(0.5)
  );
  cylinder.setRotationFromQuaternion(quaternion);
}

function createCylinder(
  joints,
  { start_joint, end_joint, start_radius, end_radius, color }
) {
  const geometry = new THREE.CylinderGeometry(
    start_radius,
    end_radius,
    getCylinderLength(joints[start_joint].position, joints[end_joint].position),
    32
  );
  const material = new THREE.MeshLambertMaterial({ color });
  const cylinder = new THREE.Mesh(geometry, material);

  updateCylinderPosition(
    cylinder,
    joints[start_joint].position,
    joints[end_joint].position
  );

  return cylinder;
}

export async function renderSequence(sequenceDir, canvas) {
  // Set Scene
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);

  // Set Camera
  const camera = new THREE.PerspectiveCamera(
    75,
    DEFAULT_ASPECT_RATIO,
    0.1,
    1000
  );
  camera.position.set(...DEFAULT_CAMERA_POSITION);
  camera.up.set(...DEFAULT_CAMERA_UP);
  camera.lookAt(...DEFAULT_LOOK_AT);
  camera.setFocalLength(DEFAULT_FOCAL_LENGTH);

  // Set controls
  const controls = new OrbitControls(camera, canvas);

  // Set Renderer
  const renderer = new THREE.WebGLRenderer({
    canvas,
    ahpha: true,
    antialias: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  // Set Light
  const spotLight1 = new THREE.SpotLight(0xffffff);
  spotLight1.position.set(10, 10, 10);
  spotLight1.penumbra = 1;
  spotLight1.decay = 0;
  spotLight1.intensity = Math.PI;
  scene.add(spotLight1);

  const spotLight2 = new THREE.SpotLight(0xffffff);
  spotLight2.position.set(-10, -10, 10);
  spotLight2.penumbra = 1;
  spotLight2.decay = 0;
  spotLight2.intensity = Math.PI;
  scene.add(spotLight2);

  // Set Floor and Grid
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(7, 7),
    new THREE.MeshBasicMaterial({
      color: 0xeeeeee,
      side: THREE.DoubleSide,
    })
  );
  scene.add(floor);
  const gridHelper = new THREE.GridHelper(7, 40, 0x888888, 0x888888); // 크기와 그리드 개수 설정
  gridHelper.rotation.x = Math.PI / 2; // 그리드를 수평으로 회전
  scene.add(gridHelper);

  // Load a human and objects
  const [
    objectColors,
    objectInScene,
    jointPositions,
    objectTransformations,
    cameraTransforms,
    cameraIntrinsics,
  ] = await Promise.all([
    loadJSON(`./color.json`),
    loadJSON(`../data/seq/${sequenceDir}/object_in_scene.json`),
    loadJSON(`../data/seq/${sequenceDir}/joint_positions.json`),
    loadJSON(`../data/seq/${sequenceDir}/object_transformations.json`),
    loadJSON(`../data/seq/${sequenceDir}/floored_scaled_transform.json`),
    loadJSON(`../data/seq/${sequenceDir}/undistorted_intrinsics.json`),
  ]);

  console.log("load joints and transformations complete");

  const objects = (
    await Promise.all(
      Object.keys(objectInScene)
        .filter((objectName) => objectInScene[objectName])
        // .filter((_, index) => index < 0)
        .map(async (objectName) => {
          // const [baseSimplified, part1Simplified, part2Simplified] =
          //   await Promise.all([
          //     loadObj(`../../objects/${objectName}/base_simplified.obj`),
          //     loadObj(`../../objects/${objectName}/part1_simplified.obj`),
          //     loadObj(`../../objects/${objectName}/part2_simplified.obj`),
          //   ]);

          const [base, part1, part2] = await Promise.all([
            loadObj(`../data/scan/${objectName}/simplified/base.obj`),
            loadObj(`../data/scan/${objectName}/simplified/part1.obj`),
            loadObj(`../data/scan/${objectName}/simplified/part2.obj`),
          ]);

          return [
            { partName: "base", object: base, objectName },
            { partName: "part1", object: part1, objectName },
            { partName: "part2", object: part2, objectName },
          ];
        })
    )
  )
    .flat()
    .filter(({ object }) => object !== null);

  console.log("load objects complete");

  // Set Joints
  const joints = jointPositions[0].map((coord, index) => {
    const radius = getJointRadius(index);

    const color = index >= 23 ? HAND_COLOR : BODY_COLOR;

    const joint = new THREE.Mesh(
      new THREE.SphereGeometry(radius, 20, 20),
      new THREE.MeshMatcapMaterial({ color })
    );
    joint.position.set(...coord);

    scene.add(joint);
    return joint;
  });

  // Set Cylinders
  const cylinders = BODY_CYLINDER_DATA.map((data) => ({
    ...data,
    color: BODY_COLOR,
  }))
    .concat(
      HAND_CYLINDER_DATA.map((data) => ({
        ...data,
        start_radius: HAND_CYLINDER_RADIUS,
        end_radius: HAND_CYLINDER_RADIUS,
        color: HAND_COLOR,
      }))
    )
    .map((data) => {
      const cylinder = createCylinder(joints, data);
      scene.add(cylinder);
      return { data, cylinder };
    });

  // Set Objects
  objects.forEach(({ partName, object, objectName }) => {
    object.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.material = new THREE.MeshLambertMaterial({
          color: convertRgbToHex(...objectColors[objectName][partName]),
        });
      }
    });
    const initialTransformation =
      objectTransformations[0][`${objectName}_${partName}`];
    if (initialTransformation) {
      object.applyMatrix4(new THREE.Matrix4(...initialTransformation.flat()));
    }
    object.matrixAutoUpdate = false;
    scene.add(object);
  });

  // Set camera phyramids

  const cameraParams = cameraTransforms.map((cameraPosition, cameraIndex) => {
    const cameraIntrinsic =
      cameraIntrinsics[String(cameraIndex + 1)]["Intrinsics"];
    const scaleFactor = 0.00005;
    const focalLength = cameraIntrinsic[0];
    const width = cameraIntrinsic[2];
    const height = cameraIntrinsic[5];

    const verticesCam = [
      [0, 0, 0],
      [width * -1, height, focalLength],
      [width * -1, height * -1, focalLength],
      [width, height * -1, focalLength],
      [width, height, focalLength],
      [0, 0, focalLength],
    ].map((row) => row.map((value) => value * scaleFactor));

    const inverse = new THREE.Matrix4(...cameraPosition.flat())
      .transpose()
      .toArray();

    const rotationMatrix = new THREE.Matrix3().set(
      inverse[0],
      inverse[1],
      inverse[2],
      inverse[4],
      inverse[5],
      inverse[6],
      inverse[8],
      inverse[9],
      inverse[10]
    );

    const translation = new THREE.Vector3().set(
      inverse[3],
      inverse[7],
      inverse[11]
    );

    const verticesWorld = verticesCam.map((vertex) =>
      new THREE.Vector3(...vertex).applyMatrix3(rotationMatrix).add(translation)
    );

    [
      [0, 1],
      [0, 2],
      [0, 3],
      [0, 4],
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 1],
    ].forEach(([index1, index2]) => {
      const vertex1 = verticesWorld[index1];
      const vertex2 = verticesWorld[index2];
      const geometry = new THREE.CylinderGeometry(
        0.001,
        0.001,
        getCylinderLength(vertex1, vertex2),
        32
      );
      const material = new THREE.MeshLambertMaterial({ color: 0x000000 });
      const cylinder = new THREE.Mesh(geometry, material);

      updateCylinderPosition(cylinder, vertex1, vertex2);

      scene.add(cylinder);
    });

    return { pyramidVertices: verticesWorld, focalLength, width, height };
  });

  const rendererWidth = canvas.offsetWidth;
  renderer.setSize(rendererWidth, rendererWidth / DEFAULT_ASPECT_RATIO);

  function updateCamera(position, lookAt, up, focalLength, aspect) {
    camera.position.copy(position);
    camera.lookAt(lookAt);
    camera.up.set(...up);
    camera.setFocalLength(focalLength);
    camera.aspect = aspect;
    camera.updateProjectionMatrix();
    renderer.setSize(rendererWidth, rendererWidth / aspect);
  }

  const onChangeCamera = (cameraLabel) => {
    if (cameraLabel === "egocentric") {
      onChangeFrameCallbacks["egocentric"] = handleEgocentricCamera;
    } else {
      onChangeFrameCallbacks["egocentric"] = undefined;
    }
    if (cameraLabel.includes("rgb")) {
      const cameraIndex = cameraLabel.split("rgb")[1] - 1;
      const {
        pyramidVertices: vertices,
        focalLength,
        width,
        height,
      } = cameraParams[cameraIndex];
      updateCamera(
        vertices[0],
        vertices[5],
        [0, 0, 1],
        focalLength / 100,
        width / height
      );
    } else if (cameraLabel === "default") {
      // default
      updateCamera(
        new THREE.Vector3(...DEFAULT_CAMERA_POSITION),
        new THREE.Vector3(...DEFAULT_LOOK_AT),
        DEFAULT_CAMERA_UP,
        DEFAULT_FOCAL_LENGTH,
        DEFAULT_ASPECT_RATIO
      );
    }
  };

  const handleEgocentricCamera = (frame, joints) => {
    const neckToHeadVector = joints[6].position
      .clone()
      .sub(joints[3].position.clone());

    const lookAt = joints[12].position
      .clone()
      .sub(joints[8].position.clone())
      .cross(neckToHeadVector)
      .multiplyScalar(10)
      .sub(neckToHeadVector.multiplyScalar(1))
      .add(joints[6].position.clone());

    joints[50].position.copy(lookAt);
    updateCamera(
      joints[6].position,
      lookAt,
      neckToHeadVector,
      DEFAULT_FOCAL_LENGTH,
      DEFAULT_ASPECT_RATIO
    );
  };

  // Set Animation
  let animateStartedTime;
  let animateStartedAt;
  let animateOffsetMs = 0;
  let prevFrame = 0;
  const onChangeFrameCallbacks = {};
  function animate(time) {
    requestAnimationFrame(animate);
    if (!time) return;
    if (animateStartedTime === undefined) {
      animateStartedTime = time;
      animateStartedAt = Date.now();
    }

    const currentFrame =
      Math.floor(((time - animateStartedTime + animateOffsetMs) / 1000) * FPS) %
      jointPositions.length;

    if (prevFrame !== currentFrame) {
      prevFrame = currentFrame;
      joints.forEach((joint, index) => {
        const coord = jointPositions[currentFrame][index];
        joint.position.set(...coord);
      });

      cylinders.forEach(({ cylinder, data: { start_joint, end_joint } }) => {
        updateCylinderPosition(
          cylinder,
          joints[start_joint].position,
          joints[end_joint].position
        );
      });

      objects.forEach(({ object, partName, objectName }) => {
        const transformation =
          objectTransformations[currentFrame][`${objectName}_${partName}`];
        if (transformation) {
          object.matrix.identity();
          object.applyMatrix4(new THREE.Matrix4(...transformation.flat()));
          object.matrixWorldNeedsUpdate = true;
        }
      });

      Object.values(onChangeFrameCallbacks).forEach((callback) =>
        callback?.(currentFrame, joints)
      );
    }

    renderer.render(scene, camera);
  }
  animate();

  return {
    renderer,
    totalFrames: jointPositions.length,
    onChangeFrame: (callback) => {
      const key = Math.random().toString();
      onChangeFrameCallbacks[key] = callback;
    },
    updateCurrentFrame: (frame) => {
      animateOffsetMs = animateStartedAt - Date.now() + (frame / FPS) * 1000;
    },
    cameraLabels: [
      "default",
      "egocentric",
      ...cameraTransforms.map((_, index) => `rgb${index + 1}`),
    ],
    onChangeCamera,
  };
}
