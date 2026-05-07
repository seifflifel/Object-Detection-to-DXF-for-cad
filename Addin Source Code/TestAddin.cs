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
using System.Globalization;



namespace TestAddin
{
    [ComVisible(true), Guid("4A1E4A4C-E0F3-498D-8BCA-4A51B21DDFE0")]
    [AutoRegister("TracingAddin", "Automaticaly Trace and Scale objects")]
    public class TestAddin : SwAddInEx
    {
        [Title("FastFlow")]
        [Description("This add-in provides commands to run a Python script, insert an image, and scale & offset sketch entities & provide shortcuts for a faster flow (inserting and moving and substracting parts)")]
        //[Icon(typeof(Resources),nameof(Resources.TraceAddinicon))]
        private enum Commands_e
        {
            [Title("CreateBox")]
            [Description("Create initial box")]
            [Icon(typeof(Resources), nameof(Resources.BoxIcon))]
            [CommandItemInfo(true, true, swWorkspaceTypes_e.Part, true)]
            CreateBox,


            [Title("Run Python")]
            [Description("Runs a Python script to process data.")]
            [Icon(typeof(Resources), nameof(Resources.Python))]
            [CommandItemInfo(true, true, swWorkspaceTypes_e.Part, true)]
            RunPython,

            [Title("Insert Image")]
            [Description("Inserts an image into the active sketch.")]
            [Icon(typeof(Resources), nameof(Resources.mask))]
            [CommandItemInfo(true, true, swWorkspaceTypes_e.Part, true)]
            InsertImage,

            [Title("Scale & Offset")]
            [Description("Scales and offsets sketch entities in the active sketch.")]
            //[Icon(typeof(Resources), nameof(Resources.ScaleAndOffsetIcon))]
            [CommandItemInfo(true, true, swWorkspaceTypes_e.Part, true)]
            ScaleAndOffset,

            [Title("Cut")]
            [Description("Cut the sketch")]
            [Icon(typeof(Resources), nameof(Resources.cut))]
            [CommandItemInfo(true, true, swWorkspaceTypes_e.Part, true)]
            Cut,


            [Title("Insert Lock")]
            [Description("Insert left or right lock")]
            [Icon(typeof(Resources), nameof(Resources.Lock))]
            [CommandItemInfo(true, true, swWorkspaceTypes_e.Part, true)]
            InsertLock,

            [Title("Insert Sensor")]
            [Description("Insert Sensor Place to Substract")]
            [Icon(typeof(Resources), nameof(Resources.Sensor))]
            [CommandItemInfo(true, true, swWorkspaceTypes_e.Part, true)]
            InsertSensor,

            [Title("Insert Sensor Cover")]
            [Description("Insert Sensor Cover")]
            [Icon(typeof(Resources), nameof(Resources.SensorCover))]
            [CommandItemInfo(true, true, swWorkspaceTypes_e.Part, true)]
            InsertSensorCover,

        }

        // UI Classes

        private PropertyManagerPageEx<PropretyPageHandler, Data> m_DataPage;
        private Data m_data;
        
        private PropertyManagerPageEx<PropretyPageHandler, Data2> m_DataPage2;
        private Data2 m_data2;

        private PropertyManagerPageEx<PropretyPageHandler, Depth> m_DepthPage;
        private Depth m_data3;

        private PropertyManagerPageEx<PropretyPageHandler, LockPage> m_LockPage;
        private LockPage m_data4;

        private PropertyManagerPageEx<PropretyPageHandler, SensorPage> m_SensorPage;
        private SensorPage m_data5;


        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

            m_LockPage = new PropertyManagerPageEx<PropretyPageHandler, LockPage>(App);
            m_LockPage.Handler.Closed += OnLockPageClosedLock;
            m_data4 = new LockPage()
            {
                Option = LockPage.Options_e.Option1, // Default to left lock
            };


            m_SensorPage = new PropertyManagerPageEx<PropretyPageHandler, SensorPage>(App);
            m_SensorPage.Handler.Closed += OnSensorPageClosed;
            m_data5 = new SensorPage()
            {
                Main = null, // This will be set by the user in the UI
                //Body = null, // This will be set by the user in the UI
            };

            return true;

            

        }

        // UI PAGES
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

        private void OnLockPageClosedLock(swPropertyManagerPageCloseReasons_e reason)
        {
            if (reason == swPropertyManagerPageCloseReasons_e.swPropertyManagerPageClose_Okay)
            {
                InsertLock(m_data4.Option);
            }
        }

        private void OnSensorPageClosed(swPropertyManagerPageCloseReasons_e reason)
        {
            if (reason == swPropertyManagerPageCloseReasons_e.swPropertyManagerPageClose_Okay)
            {
                InsertSensor(m_data5.Main);
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

                case Commands_e.Cut:
                    m_DepthPage.Show(m_data3);
                    break;

                case Commands_e.InsertLock:
                    m_LockPage.Show(m_data4);
                    break;
                case Commands_e.InsertSensor:
                    m_SensorPage.Show(m_data5);
                    break;

                case Commands_e.InsertSensorCover:
                    InsertSensorCover();
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


            // Select the Front
            bool selected = doc.Extension.SelectByID2("front plane", "PLANE", 0, 0, 0, false, 0, null, 0);  //french modification 
            if (!selected)
            {
                App.SendMsgToUser("Could not select Front Plane.");
                return;
            }


            doc.IActiveView.EnableGraphicsUpdate = false; // Disable graphics update for performance
            doc.FeatureManager.EnableFeatureTree = false; // Disable feature tree updates for performance
            try
            {
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

                    doc.ViewZoomtofit2(); // Zoom to fit after creation
                }
            }
            finally
            {

                doc.IActiveView.EnableGraphicsUpdate = true; // Disable graphics update for performance
                doc.FeatureManager.EnableFeatureTree = true; // Disable feature tree updates for performance
            }
        }


        private void RunPython()
        {
            string pythonExe = @"C:\CP files\venv\Scripts\python.exe";
            string scriptPath = @"C:\CP files\process_main.py";

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

            doc.Extension.SelectByID2("Front Plane", "PLANE", 0, 0, 0, false, 0, null, 0); //french modification 
            doc.SketchManager.InsertSketch(true);

            string imagePath = @"C:\CP files\images\mask.png";
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
            reference.Select4(false, null);                   // Select it
            doc.SketchManager.InsertSketch(true);
            var sketch = sm.ActiveSketch;

            if (sketch == null)
            { 
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

            string path = @"C:\CP files\width_scale.txt";
            double scale = 1;
            if (File.Exists(path))
            {
                string content = File.ReadAllText(path).Trim();

                if (double.TryParse(content, NumberStyles.Any, CultureInfo.InvariantCulture, out double parsedValue))

                {
                    scale = parsedValue;
                }
                else
                {
                    App.SendMsgToUser($"Invalid scale value in file.'{content}'");
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

        private void InsertLock(LockPage.Options_e optionenum )
        {
            var model = App.IActiveDoc2 as ModelDoc2;
            var partDoc = model as PartDoc;
            var path_left = @"C:\CP files\SolidParts\left locker block.SLDPRT";
            var path_right = @"C:\CP files\SolidParts\right locker block.SLDPRT";
            switch (optionenum)
            {
                case LockPage.Options_e.Option1:
                    partDoc.InsertPart3(path_left, 3, "Défaut");
                    model.Extension.SelectByID2("<left locker block>-<Cut-Extrude4>", "SOLIDBODY", 0, 0, 0, false, 1, null, 0);
                    Feature feat = model.FeatureManager.InsertMoveCopyBody2(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 1) as Feature;
                    model.Extension.SelectByID2(feat.Name, "BODYFEATURE", 0, 0, 0, false, 0, null, 0);
                    break;
                case LockPage.Options_e.Option2:
                    partDoc.InsertPart3(path_right, 3, "Défaut");
                    model.Extension.SelectByID2("<right locker block>-<Cut-Extrude4>", "SOLIDBODY", 0, 0, 0, false, 1, null, 0); //change 
                    model.FeatureManager.InsertMoveCopyBody2(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 1);
                    break;
                default:
                    App.SendMsgToUser("Invalid Path.");
                    break;
            }
        }

        private void InsertSensor(IBody2 Main)
        {
            var path = @"C:\CP files\SolidParts\Sensor Placeholder Subtract.SLDPRT";
            var model = App.IActiveDoc2;
            var partDoc = model as PartDoc;
            if (File.Exists(path))
            {
                partDoc.InsertPart3(path, 3, "Default");
                model.Extension.SelectByID2("Sensor Placeholder Subtract", "BODYFEATURE", 0, 0, 0, false, 1, null, 0);
                model.FeatureManager.InsertMoveCopyBody2(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 1);
                Main.Select(false,1);
                model.Extension.SelectByID2("<Sensor Placeholder Subtract>-<Boss-Extrude9>", "SOLIDBODY", 2.96244286658975E-03, 6.00970229277209E-03, 0, true, 2, null, 0);
                model.FeatureManager.InsertCombineFeature(15902, null, null);
            }
            else
            {
                App.SendMsgToUser("Sensor file not found.");
            }
        }

        private void InsertSensorCover()
        {
            var path = @"C:\CP files\SolidParts\Sensor Cover.SLDPRT";
            var model = App.IActiveDoc2;
            var partDoc = model as PartDoc;
            if (File.Exists(path))
            {
                partDoc.InsertPart3(path, 3, "Default");
                model.Extension.SelectByID2("<Sensor Cover>-<Sensor Cover.STEP<1>>", "SOLIDBODY", 0, 0, 0, false, 1, null, 0);
                model.FeatureManager.InsertMoveCopyBody2(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 1);
            }
            else
            {
                App.SendMsgToUser("Sensor Cover file not found.");
            }
        }
    }


}
