/*
 * Bing AI(2023/02/16版)にお願いした内容
 * 
 * Webカメラから画像を取り込んで、その中に写っている人形の顔の向きを、角度で表示するプログラムをC#言語で書いて。
 * ただしOpenCvSharpは使っていいけど、Dlibは使わないで。
 */
using System;
using System.Collections.Generic;
using System.Diagnostics.Eventing.Reader;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

// OpenCvSharpをインポートする
using OpenCvSharp;

namespace MiniFigAngleCV
{
    // プログラムのクラスを定義する
    class Program
    {
        // メインメソッドを定義する
        static void Main(string[] args)
        {
            // Webカメラからビデオキャプチャを作成する
            VideoCapture capture = new VideoCapture(0);

            // 顔検出用のカスケード分類器を作成する
            CascadeClassifier faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");

            // 顔の向きを推定するための3Dモデルの点を定義する
            Point3f[] modelPoints = new Point3f[]
            {
            new Point3f(0.0f, 0.0f, 0.0f), // 鼻の先
            new Point3f(0.0f, -330.0f, -65.0f), // 顎の先
            new Point3f(-225.0f, 170.0f, -135.0f), // 左目の左端
            new Point3f(225.0f, 170.0f, -135.0f), // 右目の右端
            new Point3f(-150.0f, -150.0f, -125.0f), // 口の左端
            new Point3f(150.0f, -150.0f, -125.0f) // 口の右端
            };

            // メインループを開始する
            while (true)
            {
                // キャプチャからフレームを取得する
                Mat frame = capture.RetrieveMat();

                // フレームが空でないことを確認する
                if (!frame.Empty())
                {
                    // フレームをグレースケールに変換する
                    Mat gray = frame.CvtColor(ColorConversionCodes.BGR2GRAY);

                    // グレースケール画像から顔を検出する
                    Rect[] faces = faceCascade.DetectMultiScale(gray);

                    // 検出された顔に対してループする
                    foreach (Rect face in faces)
                    {
                        // 顔の領域を切り出す
                        Mat faceROI = gray[face];

                        // 顔の領域から目を検出する
                        Point[] eyes = DetectEyes(faceROI);

                        // 目が2つ検出された場合
                        if (eyes.Length == 2)
                        {
                            // 顔の向きを推定するための2D画像の点を定義する
                            Point2f[] imagePoints = new Point2f[]
                            {
                            new Point2f(face.X + face.Width / 2, face.Y + face.Height / 2), // 鼻の先
                            new Point2f(face.X + face.Width / 2, face.Y + face.Height), // 顎の先
                            new Point2f(face.X + eyes[0].X, face.Y + eyes[0].Y), // 左目の中心
                            new Point2f(face.X + eyes[1].X, face.Y + eyes[1].Y), // 右目の中心
                            new Point2f(face.X + face.Width / 4, face.Y + face.Height * 3 / 4), // 口の左端
                            new Point2f(face.X + face.Width * 3 / 4, face.Y + face.Height * 3 / 4) // 口の右端
                        };

                            // カメラの内部パラメータを定義する
                            double focalLength = frame.Cols; // 焦点距離は画像の幅と仮定する
                            Point2d center = new Point2d(frame.Cols / 2, frame.Rows / 2); // 中心は画像の中心と仮定する
                            Mat cameraMatrix = new Mat(3, 3, MatType.CV_64FC1); // カメラ行列を作成する
                            cameraMatrix.Set(0, 0, focalLength); // (0, 0)要素に焦点距離をセットする
                            cameraMatrix.Set(0, 2, center.X); // (0, 2)要素に中心のX座標をセットする
                            cameraMatrix.Set(1, 1, focalLength); // (1, 1)要素に焦点距離をセットする
                            cameraMatrix.Set(1, 2, center.Y); // (1, 2)要素に中心のY座標をセットする
                            cameraMatrix.Set(2, 2, 1.0); // (2, 2)要素に1.0をセットする

                            // 歪み係数を定義する
                            Mat distCoeffs = new Mat(4, 1, MatType.CV_64FC1); // 歪み係数を作成する
                            distCoeffs.SetTo(0.0); // 歪み係数を0.0にセットする

                            // SolvePnP関数を使って顔の向きを推定する
                            Mat rvec = new Mat(); // 回転ベクトルを格納する行列を作成する
                            Mat tvec = new Mat(); // 並進ベクトルを格納する行列を作成する

                            /// 元ネタでは
                            /// SolvePnP(modelPoints, imagePoints, ...
                            /// となっており、型が合わなかったので
                            /// InputArray型に変換してあげる
                            
                            // Point3f[] を cv::Mat に変換
                            Mat mat1 = new Mat(modelPoints.Length, 1, MatType.CV_32FC3, modelPoints);
                            // cv::Mat を InputArray に変換
                            InputArray input = mat1;
                            Mat mat2 = new Mat(imagePoints.Length, 1, MatType.CV_32FC2, imagePoints);
                            InputArray input2 = mat2;

                            Cv2.SolvePnP(input, input2, cameraMatrix, distCoeffs, rvec, tvec); // SolvePnP関数を実行する

                            // 回転ベクトルを回転行列に変換する
                            Mat rmat = new Mat(); // 回転行列を格納する行列を作成する
                            Cv2.Rodrigues(rvec, rmat); // Rodrigues関数を実行する

                            // 回転行列からオイラー角を計算する
                            double[] euler = RotationMatrixToEulerAngles(rmat); // オイラー角を格納する配列を作成する
                            double pitch = euler[0]; // ピッチ角を取得する
                            double yaw = euler[1]; // ヨー角を取得する
                            double roll = euler[2]; // ロール角を取得する

                            // 顔の向きを角度で表示する
                            Cv2.PutText(frame, $"Pitch: {pitch:F1} deg", new Point(face.X, face.Y - 30), HersheyFonts.HersheySimplex, 0.8, Scalar.Red, 2); // ピッチ角を表示する
                            Cv2.PutText(frame, $"Yaw: {yaw:F1} deg", new Point(face.X, face.Y - 10), HersheyFonts.HersheySimplex, 0.8, Scalar.Green, 2); // ヨー角を表示する
                            Cv2.PutText(frame, $"Roll: {roll:F1} deg", new Point(face.X, face.Y + 10), HersheyFonts.HersheySimplex, 0.8, Scalar.Blue, 2); // ロール角を表示する
                        }
                    }

                    // フレームを表示する
                    Cv2.ImShow("Face Orientation", frame);
                }

                // キー入力を待つ
                int key = Cv2.WaitKey(1);

                // Escキーが押された場合
                if (key == 27)
                {
                    // メインループを終了する
                    break;
                }
            }

            // ビデオキャプチャを解放する
            capture.Release();

            // ウィンドウを破棄する
            Cv2.DestroyAllWindows();
        }

        // 目を検出するメソッドを定義する
        static Point[] DetectEyes(Mat faceROI)
        {
            // 目検出用のカスケード分類器を作成する
            CascadeClassifier eyeCascade = new CascadeClassifier("haarcascade_eye.xml");

            // 顔の領域から目を検出する
            Rect[] eyes = eyeCascade.DetectMultiScale(faceROI);

            // 目の中心の座標を格納するリストを作成する
            List<Point> eyeCenters = new List<Point>();

            // 検出された目に対してループする
            foreach (Rect eye in eyes)
            {
                // 目の中心の座標を計算する
                Point eyeCenter = new Point(eye.X + eye.Width / 2, eye.Y + eye.Height / 2);

                // 目の中心の座標をリストに追加する
                eyeCenters.Add(eyeCenter);
            }

            // 目の中心の座標のリストを配列に変換して返す
            return eyeCenters.ToArray();
        }

        // 回転行列からオイラー角を計算するメソッドを定義する
        static double[] RotationMatrixToEulerAngles(Mat rmat)
        {
            // 回転行列の要素を取得する
            double m00 = rmat.At<double>(0, 0);
            double m01 = rmat.At<double>(0, 1);
            double m02 = rmat.At<double>(0, 2);
            double m10 = rmat.At<double>(1, 0);
            double m11 = rmat.At<double>(1, 1);
            double m12 = rmat.At<double>(1, 2);
            double m20 = rmat.At<double>(2, 0);
            double m21 = rmat.At<double>(2, 1);
            double m22 = rmat.At<double>(2, 2);

            // オイラー角を格納する配列を作成する
            double[] euler = new double[3];

            // ピッチ角を計算する
            double sy = Math.Sqrt(m00 * m00 + m10 * m10);
            bool singular = sy < 1e-6;
            if (!singular)
            {
                euler[0] = Math.Atan2(m21, m22) * 180 / Math.PI; // ラジアンから度に変換する
                euler[1] = Math.Atan2(-m20, sy) * 180 / Math.PI; // ラジアンから度に変換する
                euler[2] = Math.Atan2(m10, m00) * 180 / Math.PI; // ラジアンから度に変換する
            }
            else
            {
                euler[0] = Math.Atan2(-m12, m11) * 180 / Math.PI; // ラジアンから度に変換する
                euler[1] = Math.Atan2(-m20, sy) * 180 / Math.PI; // ラジアンから度に変換する
                euler[2] = 0;
            }

            // オイラー角の配列を返す
            return euler;
        }
    }
}