using System;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using CodeStack.SwEx.AddIn;
using CodeStack.SwEx.AddIn.Attributes;
using CodeStack.SwEx.Common.Attributes;
using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;
using System.Diagnostics;
using System.ComponentModel;
using TestAddin.Properties;
using CodeStack.SwEx.AddIn.Enums;


namespace TestAddin
{
    [ComVisible(true),Guid("4A1E4A4C-E0F3-498D-8BCA-4A51B21DDFE0")]
    [AutoRegister("TracingAddin","Automaticaly Trace and Scale objects")]
    public class TestAddin : SwAddInEx
    {
        [Title("TraceAddin")]
        [Description("This add-in provides commands to run a Python script, insert an image, and scale & offset sketch entities.")]
        //[Icon(typeof(Resources),nameof(Resources.TraceAddinicon))]
        private enum Commands_e
        {
            [Title("Run Python")]
            [Description("Runs a Python script to process data.")]
            //[Icon(typeof(Resources), nameof(Resources.RunPythonIcon))]
            [CommandItemInfo(true,true,swWorkspaceTypes_e.Part,true)]
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
            ScaleAndOffset
        }

        public override bool OnConnect()
        {
            AddCommandGroup<Commands_e>(OnButtonClick);
            return true;
        }

        private void OnButtonClick(Commands_e command)
        {
            switch (command)
            {
                case Commands_e.RunPython:
                    RunPython();
                    break;

                case Commands_e.InsertImage:
                    InsertImage();
                    break;

                case Commands_e.ScaleAndOffset:
                    ScaleAndOffset();
                    break;
            }
        }

        private void RunPython()
        {
            string pythonExe = @"C:\Users\Seifo\AppData\Local\Programs\Python\Python313\python.exe";
            string scriptPath = @"C:\Users\Seifo\Documents\Stage ete 2025\full_with_ArUco_calib.py";

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

        private void ScaleAndOffset()
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
                App.SendMsgToUser("No active sketch.");
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

            doc.Extension.ScaleOrCopy(false, 1, 0, 0, 0, 0.083);

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

            double offset = 0.3 / 1000.0;
            bool success = sm.SketchOffset2(offset, false, true, 1, 1, true);

            if (!success)
                App.SendMsgToUser("Offset failed.");


            doc.ViewZoomtofit2();
        }
    }
    }