using CodeStack.SwEx.AddIn;
using CodeStack.SwEx.AddIn.Attributes;
using CodeStack.SwEx.AddIn.Enums;
using CodeStack.SwEx.Common.Attributes;
using CodeStack.SwEx.PMPage;
using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using TestAddin.Properties;
using static System.Net.WebRequestMethods;
using File = System.IO.File;


namespace TestAddin
{
    [ComVisible(true), Guid("4A1E4A4C-E0F3-498D-8BCA-4A51B21DDFE0")]
    [AutoRegister("TracingAddin", "Automaticaly Trace and Scale objects")]
    public class TestAddin : SwAddInEx
    {
        [Title("TraceAddin")]
        [Description("This add-in provides commands to run a Python script, insert an image, and scale & offset sketch entities.")]
        //[Icon(typeof(Resources),nameof(Resources.TraceAddinicon))]
        private enum Commands_e
        {
            [Title("CreateBox")]
            [Description("Create initial box")]
            //[Icon(typeof(Resources), nameof(Resources.ScaleAndOffsetIcon))]
            [CommandItemInfo(true, true, swWorkspaceTypes_e.Part, true)]
            CreateBox,


            [Title("Run Python")]
            [Description("Runs a Python script to process data.")]
            //[Icon(typeof(Resources), nameof(Resources.RunPythonIcon))]
            [CommandItemInfo(true, true, swWorkspaceTypes_e.Part, true)]
            RunPython,

            [Title("Insert Image")]
            [Description("Inserts an image into the active sketch.")]
            //[Icon(typeof(Resources), nameof(Resources.InsertImageIcon))]
            [CommandItemInfo(true, true, swWorkspaceTypes_e.Part, true)]
            InsertImage,

            [Title("Scale & Offset")]
            [Description("Scales and offsets sketch entities in the active sketch.")]
            //[Icon(typeof(Resources), nameof(Resources.ScaleAndOffsetIcon))]
            [CommandItemInfo(true, true, swWorkspaceTypes_e.Part, true)]
            ScaleAndOffset,

            //[Title("Extrude")]
            //[Description("Extrude from contour")]
            //[CommandItemInfo(true, true, swWorkspaceTypes_e.Part, true)]
            //ExtrudeContour,

            [Title("Insert Lock")]
            [Description("Insert left or right lock")]
            [CommandItemInfo(true, true, swWorkspaceTypes_e.Part, true)]
            InsertLock,

            [Title("Insert Sensor")]
            [Description("Insert Sensor Place to Substract")]
            [CommandItemInfo(true, true, swWorkspaceTypes_e.Part, true)]
            InsertSensor,

            [Title("Cut")]
            [Description("Cut the sketch")]
            [CommandItemInfo(true, true, swWorkspaceTypes_e.Part, true)]
            Cut,

        }


        private PropertyManagerPageEx<PropretyPageHandler, Data> m_DataPage;
        private Data m_data;
        
        private PropertyManagerPageEx<PropretyPageHandler, Data2> m_DataPage2;
        private Data2 m_data2;

        private PropertyManagerPageEx<PropretyPageHandler, Depth> m_DepthPage;
        private Depth m_data3;


        public override bool OnConnect()
        {
            AddCommandGroup<Commands_e>(OnButtonClick);
            m_DataPage= new PropertyManagerPageEx<PropretyPageHandler, Data>(App);
            m_DataPage.Handler.Closed += OnDataPageClosed;
            m_data = new Data()
            {
                Offset = 0.0003,  // Default offset in meters

            };

            m_DataPage2 = new PropertyManagerPageEx<PropretyPageHandler, Data2>(App);
            m_DataPage2.Handler.Closed += OnDataPageClosedBox;
            m_data2 = new Data2()
            {
                width = 0.050,  // mm
                length = 0.050, // mm
                height = 0.009, // mm
            };


            m_DepthPage = new PropertyManagerPageEx<PropretyPageHandler, Depth>(App);
            m_DepthPage.Handler.Closed += OnDataPageCloseDepth;
            m_data3 = new Depth()
            {   
                Reference = null, // This will be set by the user in the UI
                depth = 0.008,
            };
            return true;
        }

        private void OnDataPageClosed(swPropertyManagerPageCloseReasons_e reason)
        {
            if(reason == swPropertyManagerPageCloseReasons_e.swPropertyManagerPageClose_Okay)
            {
                ScaleAndOffset(m_data.Reference,m_data.Offset);
            }

        }
        private void OnDataPageClosedBox(swPropertyManagerPageCloseReasons_e reason)
        {
            if (reason == swPropertyManagerPageCloseReasons_e.swPropertyManagerPageClose_Okay)
            {
                CreateBox(m_data2.width,m_data2.length,m_data2.height);
            }

        }

        private void OnDataPageCloseDepth(swPropertyManagerPageCloseReasons_e reason)
        {
            if (reason == swPropertyManagerPageCloseReasons_e.swPropertyManagerPageClose_Okay)
            {
                Cut(m_data3.Reference,m_data3.depth);
            }

        }



        private void OnButtonClick(Commands_e command)
        {
            switch (command)
            {
                case Commands_e.CreateBox:
                    m_DataPage2.Show(m_data2);
                    break;

                case Commands_e.RunPython:
                    RunPython();
                    
                    break;

                case Commands_e.InsertImage:
                    InsertImage();
                    break;

                case Commands_e.ScaleAndOffset:
                    m_DataPage.Show(m_data);
                    break;
                    
                //case Commands_e.ExtrudeContour:
                 //   ExtrudeContour();
                  //  break;
                case Commands_e.InsertLock:
                    InsertLock(0);
                    break;
                case Commands_e.InsertSensor:
                    InsertSensor();
                    break;

                case Commands_e.Cut:
                    m_DepthPage.Show(m_data3);
                    break;
            }
        }

        private void CreateBox(double width, double length, double height)
        {
            var doc = App.IActiveDoc2 as ModelDoc2;

            // Create a new part if none is open
            if (doc == null)
            {
                App.NewPart();
                doc = App.IActiveDoc2 as ModelDoc2;
            }


            // Select the Front Plane
            bool selected = doc.Extension.SelectByID2("Front Plane", "PLANE", 0, 0, 0, false, 0, null, 0);
            if (!selected)
            {
                App.SendMsgToUser("Could not select Front Plane.");
                return;
            }

            // Start sketch
            var skMgr = doc.SketchManager;
            skMgr.InsertSketch(true);
            skMgr.AddToDB = true;

            // Draw center rectangle
            skMgr.CreateCenterRectangle(0, 0, 0, width / 2, length / 2, 0);

            skMgr.AddToDB = false;
            skMgr.InsertSketch(true); // Exit sketch

            // Select the last created sketch
            bool sketchSelected = doc.Extension.SelectByID2("", "SKETCH", 0, 0, 0, false, 0, null, 0);
            if (!sketchSelected)
            {
                App.SendMsgToUser("Failed to select the sketch.");
                return;
            }

            // Extrude the sketch
            var featMgr = doc.FeatureManager;
            var feat = featMgr.FeatureExtrusion2(
            true,   // Sd (Solid body)
            false,  // Flip
            true,  // Dir (Direction)
            (int)swEndConditions_e.swEndCondBlind, // T1
            0,      // T2 (ignored)
            height, // D1 (Depth1)
            0,      // D2 (Depth2)
            false, false, false, false,  // Draft check and direction
            0, 0,                        // Draft angles
            false, false,               // OffsetReverse1/2
            false, false,               // TranslateSurface1/2
            true,                      // Merge
            false,                     // UseFeatScope
            false,                     // UseAutoSelect
            0,                         // T0 (Sketch direction)
            0.0,                       // StartOffset (set to 0.0)
            false                      // FlipStartOffset
            );



            if (feat == null)
            {
                App.SendMsgToUser("Extrude failed.");
            }
            else
            {
                App.SendMsgToUser("Box created.");
                doc.ViewZoomtofit2(); // Zoom to fit after creation
            }
        }


        private void RunPython()
        {
            string pythonExe = @"C:\Users\Seifo\AppData\Local\Programs\Python\Python313\python.exe";
            string scriptPath = @"C:\Users\Seifo\Documents\Stage ete 2025\full rempved z height.py";

            try
            {
                var psi = new ProcessStartInfo(pythonExe, $"\"{scriptPath}\"")
                {
                    UseShellExecute = false
                };

                var process = Process.Start(psi);
                process.WaitForExit();

                App.SendMsgToUser("Python script finished.");
            }
            catch (Exception ex)
            {
                App.SendMsgToUser("Error running Python:\n" + ex.Message);
            }
        }

        private void InsertImage()
        {
            var doc = App.IActiveDoc2 as ModelDoc2;

            if (doc == null)
            {
                App.NewPart();
                doc = App.IActiveDoc2 as ModelDoc2;
            }

            doc.Extension.SelectByID2("Front Plane", "PLANE", 0, 0, 0, false, 0, null, 0);
            doc.SketchManager.InsertSketch(true);

            string imagePath = @"C:\Users\Seifo\Documents\Stage ete 2025\mask.png";
            doc.SketchManager.InsertSketchPicture(imagePath);

            doc.ViewZoomtofit2();
        }

        private void ScaleAndOffset(IEntity reference, double offset)
        {
            var doc = App.IActiveDoc2 as ModelDoc2;

            if (doc == null)
            {
                App.SendMsgToUser("No document open.");
                return;
            }

            var sm = doc.SketchManager;
            var sketch = sm.ActiveSketch;

            if (sketch == null)
            {
                reference.Select4(false, null);                   // Select it
                doc.SketchManager.InsertSketch(true);
                sketch = sm.ActiveSketch;
                return;
            }

            doc.ClearSelection2(true);

            var segs = sketch.GetSketchSegments() as object[];
            if (segs != null)
            {
                foreach (SketchSegment s in segs)
                    s.Select4(true, null);
            }

            var pts = sketch.GetSketchPoints2() as object[];
            if (pts != null)
            {
                foreach (SketchPoint p in pts)
                    p.Select4(true, null);
            }

            string path = @"C:\Users\Seifo\Documents\Stage ete 2025\width_scale.txt";
            double scale = 1;
            if (File.Exists(path))
            {
                string content = File.ReadAllText(path).Trim();

                if (double.TryParse(content, out double parsedValue))
                {
                    scale = parsedValue;
                }
                else
                {
                    App.SendMsgToUser("Invalid scale value in file.");
                }
            }
            else
            {
                App.SendMsgToUser("Scale file not found.");
            }
            


            doc.Extension.ScaleOrCopy(false, 1, 0, 0, 0, scale); // should be read from text file

            doc.ClearSelection2(true);

            if (segs != null)
            {
                foreach (SketchSegment s in segs)
                    s.Select4(true, null);
            }

            if (pts != null)
            {
                foreach (SketchPoint p in pts)
                    p.Select4(true, null);
            }

            var construction = 1; // original construction or  not

            bool success = sm.SketchOffset2(offset, false, true, 1, construction, true);

            if (!success)
                App.SendMsgToUser("Offset failed.");


            doc.ViewZoomtofit2();
        }

        private void ExtrudeContour()
        {
            var doc = App.IActiveDoc2 as ModelDoc2;
            if (doc == null)
            {
                App.SendMsgToUser("No document open.");
                return;
            }

            // Select the sketch
            bool selected = doc.Extension.SelectByID2("Sketch1", "SKETCH", 0, 0, 0, false, 0, null, 0);
            if (!selected)
            {
                App.SendMsgToUser("Failed to select the sketch.");
                return;
            }

            var featMgr = doc.FeatureManager;
            var feat = featMgr.FeatureExtrusion2(
                true,   // Solid
                false,  // Flip
                false,  // Dir
                (int)swEndConditions_e.swEndCondBlind, // T1
                0,      // T2
                0.05,   // D1
                0.0,    // D2
                false, false, false, false,
                0.0, 0.0,
                false, false,
                false, false,
                true,   // Merge
                false,  // UseFeatScope
                false,  // UseAutoSelect
                0,      // Thin type (ignored)
                0.0,    // Wall thickness (ignored)
                false   // Flip thin direction (ignored)
            );

            if (feat == null)
            {
                App.SendMsgToUser("Extrude failed.");
            }
            else
            {
                App.SendMsgToUser("Extruded ring area between inner and outer contour.");
                doc.ViewZoomtofit2();
            }
        }


        private void Cut(IEntity reference, double Depth)
        {
            var doc = App.IActiveDoc2 as ModelDoc2;
            if (doc == null)
            {
                App.SendMsgToUser("No document open.");
                return;
            }

            if (reference == null || !reference.Select(false))
            {
                App.SendMsgToUser("Failed to select the sketch from reference.");
                return;
            }

            var featMgr = doc.FeatureManager;

            var feat = featMgr.FeatureCut4(
                true,   // Solid
                false,  // Flip
                false,  // Direction
                0, 0,   // T1, T2
                Depth,  // D1
                0,      // D2
                false, false, false, false,  // Draft
                0, 0,                          // Draft angles
                false, false,                 // Offset reverse
                false, false, false,          // Translate Surface
                true, true, true,             // Merge/Scope
                true,                         // Auto select
                false,                        // Thin feature
                0, 0,                         // Thin wall, direction
                false, false                  // Start offset, flip
            );

            if (feat == null)
            {
                App.SendMsgToUser("Cut failed.");
            }
            else
            {
                App.SendMsgToUser("Cut created successfully.");
                doc.ViewZoomtofit2();
            }
        }

        private void InsertLock(int choose)
        {
            var model = App.IActiveDoc2;
            var partDoc = model as PartDoc;
            var path_left = @"C:\Users\Seifo\Documents\Stage ete 2025\REF\REF\left locker block.SLDPRT";
            var path_right = @"C:\Users\Seifo\Documents\Stage ete 2025\right_lock.sldprt";
            
            if (choose == 0)
            {
                if (File.Exists(path_left))
                {
                    partDoc.InsertPart3(path_left, 3, "Défaut");
                    model.Extension.SelectByID2("<left locker block>-<Move Face1>", "SOLIDBODY", 0, 0, 0, false, 1, null, 0);
                    model.FeatureManager.InsertMoveCopyBody2(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 1);
                }
                else
                {
                    App.SendMsgToUser("Left lock file not found.");
                }
            }
            else if (choose == 1) //public bool Right { get; set; } = false;
            {
                if (File.Exists(path_right))
                {
                    partDoc.InsertPart3(path_right, 3, "Défaut");
                    model.Extension.SelectByID2("<Right Lock>-<Right Lock.STEP<1>>", "SOLIDBODY", 0, 0, 0, false, 1, null, 0); //change 
                    model.FeatureManager.InsertMoveCopyBody2(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 1);

                }
                else
                {
                    App.SendMsgToUser("Right lock file not found.");
                }
            }
            else
            {
                App.SendMsgToUser("Invalid choice for lock insertion.");
            }

            
        }

        private void InsertSensor()
        {
            var path = @"C:\Users\Seifo\Documents\Stage ete 2025\REF\REF\Sensor Placeholder Subtract.SLDPRT";
            var model = App.IActiveDoc2;
            var partDoc = model as PartDoc;
            if (File.Exists(path))
            {
                partDoc.InsertPart3(path, 3, "Default");
                model.Extension.SelectByID2("Sensor Placeholder Subtract", "BODYFEATURE", 0, 0, 0, false, 1, null, 0);
                model.FeatureManager.InsertMoveCopyBody2(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 1);
                model.Extension.SelectByID2("Boss-Extrude1", "SOLIDBODY", -5.01302419639273E-02, 2.35658730450155E-02, 0.056825346441201, false, 1, null, 0); // change this to be selected with ui
                model.Extension.SelectByID2("<Sensor Placeholder Subtract>-<Boss-Extrude9>", "SOLIDBODY", 2.96244286658975E-03, 6.00970229277209E-03, 0, true, 2, null, 0);
                model.FeatureManager.InsertCombineFeature(15902, null, null);
            }
            else
            {
                App.SendMsgToUser("Sensor file not found.");
            }
        }
    }
}
